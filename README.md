# Agent arXiv Daily

**Last Updated:** 2026-06-18 06:09:32

**Total Papers:** 91

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (4 papers)</h2></summary>

<details>
<summary><strong>Beyond Reward Engineering: A Data Recipe for Long-Context Reinforcement Learning</strong> - Xiaoyue Xu, Sikui Zhang, Xiaorong Wang, Xu Han, Chaojun Xiao - [[pdf]](https://arxiv.org/pdf/2606.18831)</summary>

**Abstract:** Long-context reasoning is an essential capability for large language models, particularly when they are deployed as autonomous agents that must reason over lengthy trajectories. Reinforcement learning (RL) has recently emerged as a dominant paradigm for improving this ability, yet existing work largely focuses on reward engineering while diverse training data remains scarce. We revisit this problem from a data-centric perspective and show that a simple yet effective data recipe alone, paired with a minimal outcome-based GRPO setup, suffices to substantially improve long-context reasoning. Our recipe targets three complementary task families -- retrieval, multi-evidence synthesis, and reasoning -- for which we construct and curate eight datasets totaling ~14K examples. Experiments on three models (Qwen3-4B/8B/30B-A3B) yield average gains of +7.2/+3.2/+6.4 points across seven long-context benchmarks, surpassing prior RL training sets. We further demonstrate that these gains transfer to agentic tasks, where continuing RL training on an agent-tuned model with our data recipe improves GAIA by +4.8 and BrowseComp by +7.0 points. We will release our datasets to facilitate future research.

**arXiv ID:** 2606.18831
</details>

<details>
<summary><strong>Correct Yourself, Keep My Trust: How Self-Correction and Social Connection Shape Credibility in Social Chatbots</strong> - Biswadeep Sen, Yi-Chieh Lee - [[pdf]](https://arxiv.org/pdf/2606.19286)</summary>

**Abstract:** When social chatbots make mistakes, and they do, how they recover determines whether users trust them again. Social chatbots are increasingly integrated into everyday life, yet they remain prone to generating convincing but inaccurate information. The social connection they build with users makes such errors particularly consequential. We conducted a between-subjects experiment (N=120) comparing three error correction strategies: a webpage retraction, self-correction by the same social chatbot, and correction by an expert chatbot. Our results reveal two key findings. First, all three strategies corrected the error equally well, but only self-correction did so without damaging the chatbot's credibility: participants rated self-correcting chatbots significantly higher in both trustworthiness and perceived expertise than chatbots whose errors were corrected by external sources. Second, the strength of the user's social connection with the chatbot, measured through social attraction and self-disclosure, significantly predicted the magnitude of belief change, but only when the chatbot corrected itself. Outsourcing corrections to an external source severed this link entirely. These findings suggest that social chatbots should correct their own mistakes rather than outsource corrections, and that investing in social connection is a functional mechanism that amplifies correction effectiveness, not merely a design feature. We discuss implications for designing chatbots that maintain long-term credibility while effectively addressing their own errors.

**arXiv ID:** 2606.19286
</details>

<details>
<summary><strong>CoreMem: Riemannian Retrieval and Fisher-Guided Distillation for Long-Term Memory in Dialogue Agents</strong> - Jiaqi Chen, Yongqin Zeng, Shaoshen Chen, Yijian Zhang, Hai-Tao Zheng, Chunxia Ma, XiuTeng Zhou - [[pdf]](https://arxiv.org/pdf/2606.18406)</summary>

**Abstract:** Personalized dialogue agents require continuous long-term memory to maintain coherent interactions across multiple sessions. However, deploying these capabilities on consumer-grade hardware (e.g., 8 GB VRAM edge devices) introduces severe memory and compute bottlenecks. Existing systems typically rely on isotropic cosine similarity for retrieval and heuristic rules for context compression. These approaches lack a unified theoretical foundation, frequently suffering from the hubness problem in high-dimensional retrieval and syntactic fragmentation during compression. To overcome these limitations, we propose CoreMem, a resource-efficient edge-cloud memory architecture fundamentally unified by information geometry. First, Riemannian retrieval replaces cosine matching with a locally adaptive Fisher-Rao metric, effectively penalizing hub memories via Mahalanobis distance with O(Ndr) Woodbury acceleration for real-time search. Second, Fisher-guided discrete token distillation (FDTD) introduces a hierarchical sentence-to-token compression mechanism. It derives sensitivity scores from Fisher information traces, providing a principled compression-KL tradeoff augmented with explicit structural syntax protection. Evaluated on the LOCOMO and LongMemEval-S benchmarks, CoreMem achieves strong accuracy improvements, yielding substantial gains in Open-domain (+4.51 pp) and Temporal (+4.17 pp) reasoning. Extensive profiling confirms that CoreMem operates seamlessly within a strict 8 GB VRAM budget, successfully bridging the gap between resource-constrained edge devices and the demand for theoretically grounded, lifelong memory agents.

**arXiv ID:** 2606.18406
</details>

<details>
<summary><strong>ResearchClawBench: A Benchmark for End-to-End Autonomous Scientific Research</strong> - Wanghan Xu, Shuo Li, Tianlin Ye, Qinglong Cao, Yixin Chen, Hengjian Gao, Yiheng Wang, Qi Li, Kun Li, Sheng Xu, Shengdu Chai, Fangchen Yu, Xiangyu Zhao, Zhangrui Zhao, Weijie Ma, Zijie Guo, Koutian Wu, Haoyu Zhou, Haoxiang Yin, Lixue Cheng, Chaofan Hu, Haoxuan Li, Lu Mi, Xuxuan Xie, Yifan Zhou, Ruizhe Chen, Zhiwang Zhou, Xingjian Guo, Yuhao Zhou, Xuming He, Shengyuan Xu, Xinyu Gu, Jiamin Wu, Mianxin Liu, Chunfeng Song, Fenghua Ling, Dongzhan Zhou, Shixiang Tang, Yuqiang Li, Mao Su, Peng Ye, Siqi Sun, Bin Wang, Xue Yang, Zhenfei Yin, Tianfan Fu, Guangtao Zhai, Wanli Ouyang, Bo Zhang, Lei Bai, Wenlong Zhang - [[pdf]](https://arxiv.org/pdf/2606.07591)</summary>

**Abstract:** AI coding agents are increasingly used for scientific work, but their end-to-end autonomous research capability remains difficult to verify. We present ResearchClawBench, a benchmark for evaluating autonomous scientific research across 40 tasks from 10 scientific domains. Each task is grounded in a real published paper, provides related literature and raw data, and hides the target paper during evaluation. Expert-curated multimodal rubrics decompose the target scientific artifacts into weighted criteria, enabling evaluation of target-paper-level re-discovery while leaving room for new discovery. We evaluate seven autonomous research (auto-research) agents under a unified protocol and seventeen native LLMs through the lightweight ResearchHarness. Current systems remain far from reliable re-discovery: the strongest autonomous agent, Claude Code, averages 21.5, and the strongest ResearchHarness LLM, Claude-Opus-4.7, averages 20.7, with an LLM frontier mean of only 26.5. Error analysis shows that failures concentrate in experimental protocol mismatch, evidence mismatch, and missing scientific core. ResearchClawBench provides a reproducible evaluation frontier for measuring progress toward autonomous scientific research.

**arXiv ID:** 2606.07591
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (19 papers)</h2></summary>

<details>
<summary><strong>NAVI-Orbital: First In-Orbit Demonstration of a Zero-Shot Vision-Language Model for Autonomous Earth Observation</strong> - Juan Manuel Delfa Victoria, Taran Cyriac John, Andrew W. Herson - [[pdf]](https://arxiv.org/pdf/2606.18271)</summary>

**Abstract:** As Earth Observation data generation outpaces downlink bandwidth and human-in-the-loop processing, a widening gap has emerged between onboard collection and actionable ground intelligence. This paper presents NAVI-Orbital, a software system deployed on a Low Earth Orbit (LEO) spacecraft. On April 16, 2026, NAVI-Orbital achieved what is, to the authors' knowledge, the first in-orbit demonstration of a vision-language model performing autonomous multi-modal inference entirely onboard. NAVI-Orbital uses a local vision-language model (Gemma 3) to classify each captured scene, produce a text description of its content and the relationships between its features, and respond to operator follow-up via natural-language dialogue. The system is re-tasked through plain-English prompts in place of conventional command sequences, and is orchestrated by a graph-based state machine (LangGraph) coordinating dedicated agents for detection and dialogue. Results across ground benchmarking (88.16% accuracy on the 7,960-image curated AID benchmark), Flatsat validation, and live in-orbit captures of newly acquired, previously unseen Earth imagery (including uncorrected YAM-9 imagery, processed onboard with hardware-accelerated GPU inference and no fine-tuning for the flight instrument) demonstrate the feasibility of running foundation models on satellite-class edge computers to invert the conventional acquire-then-downlink-everything bandwidth profile through semantic compression of Earth observations in-orbit.

**arXiv ID:** 2606.18271
</details>

<details>
<summary><strong>WorldLines: Benchmarking and Modeling Long-Horizon Stateful Embodied Agents</strong> - Yehang Zhang, Jianchong Su, Haojian Huang, Yifan Chang, Tianhao Zhou, Xinli Xu, Yingjie Xu, Yinchuan Li, Zexi Li, Ying-Cong Chen - [[pdf]](https://arxiv.org/pdf/2606.18847)</summary>

**Abstract:** To assist humans over extended periods in real homes, embodied agents must remember user routines, world states, and past interactions. Existing long-term memory benchmarks mainly evaluate language-centric retrieval and question answering, while embodied benchmarks often focus on short-horizon task execution without testing long-term memory use in dynamic environments. We introduce WorldLines, a project-driven benchmark for long-horizon embodied household assistance. It constructs temporally extended household traces with dialogues, actions, execution feedback, object and device state changes, and converts them into evidence-linked samples for Memory QA and Embodied Task Planning. We further propose ObsMem, an observer-grounded memory framework that maintains visibility-aware memories and action-native state trails for state-aware decisions. Experiments reveal persistent challenges in partial observability, overwritten world states, and translating long-term memory into embodied plans, while ObsMem offers a stronger reference architecture for this setting.

**arXiv ID:** 2606.18847
</details>

<details>
<summary><strong>TxBench-PP: Analyzing AI Agent Performance on Small-Molecule Preclinical Pharmacology</strong> - Hannah Le, Ramesh Ramasamy, Alex Urrutia, Mahsa Yazdani, Tim Proctor, Kenny Workman - [[pdf]](https://arxiv.org/pdf/2606.19245)</summary>

**Abstract:** Artificial intelligence (AI) agents promise to accelerate drug discovery by compressing interpretation and decision-making loops, but practical deployment requires trusted evaluation on realistic program decisions. We introduce TherapeuticsBench Preclinical Pharmacology (TxBench-PP), a verifiable benchmark for small-molecule preclinical pharmacology and the first focused slice of a broader TherapeuticsBench effort across drug-discovery stages and therapeutic modalities. TxBench-PP tests whether agents can recover accurate conclusions from real-world assay data rather than memorized facts from literature. The benchmark contains 100 evaluations indexed by program stage, assay type, and task structure, spanning mechanism-of-action (MoA) and pharmacodynamic (PD) reasoning, compound-target engagement, causal target validation, developability and safety, and translational efficacy. Agents receive realistic workflow snapshots, inspect files in a coding environment, and return structured answers graded deterministically. Across 16 model-harness configurations, comprising 11 models and 4,800 trajectories, no system reliably recovered preclinical pharmacology decisions. The strongest configuration, Claude Opus 4.8 / Pi, passed 59.3\% of endpoint attempts (178/300; 95\% CI, 51.1-67.6), followed by GPT-5.5 / Pi at 55.3\% (166/300; 47.0-63.6).

**arXiv ID:** 2606.19245
</details>

<details>
<summary><strong>ASTRA: A Scalable Next-Generation ATCO Training Simulator with Autonomous Simpilots</strong> - Ethan Chew, Enjia Wu, Iruss Eng Wei Yeow, Ian Weiqin Lim, Ranen Sim, Brandon Koh Ziheng, Kaleb Nim, Caden Toh Jun Yi, Wei Dong Soin, Darius Kai Keat Koh, Galen King Yu Tay, Prannaya Gupta, Jonathan Ee Fang Koong, Yong Zhi Lim - [[pdf]](https://arxiv.org/pdf/2606.18319)</summary>

**Abstract:** Air Traffic Control Operators (ATCOs) are vital in ensuring the safe, orderly, and efficient flow of air traffic, yet training capacity is constrained by reliance on specialized human trainers known as simpilots, who must role-play both pilots and ATCOs in a simulated airspace. Existing automated solutions rely on Western-centric speech models that perform poorly in Singaporean operational contexts, with off-the-shelf systems exhibiting Word Error Rates (WER) of up to 107.80% on Singaporean-accented aviation speech. We introduce ASTRA, an end-to-end training simulator that automates these simpilot roles through a pipeline that transcribes ATCO speech, interprets instructions, and generates appropriate pilot and ATCO responses using locally adapted voice models. Our fine-tuned Automatic Speech Recognition (ASR) pipeline reduces WER to 23.45%, substantially outperforming existing approaches in this domain. Beyond traffic simulation, ASTRA incorporates an AI-assisted performance evaluation framework that assesses trainee radiotelephony communications across accuracy, brevity, and completeness, achieving post-optimization scores of 91.7%, 88.2%, and 86.9%, respectively. Built on open-source foundations such as DSPy and Unsloth, this approach enables scalable, standardized ATCO assessment while reducing instructor workload.

**arXiv ID:** 2606.18319
</details>

<details>
<summary><strong>LandslideAgent with Multimodal LandslideBench: A Domain-Rule-Augmented Agent for Autonomous Landslide Identification and Analysis</strong> - Chengfu Liu, Dongyang Hou, Junwu Xiang, Cheng Yang, Xuezhi Cui, Zeyuan Wang, Liangtian Liu, Zelang Miao - [[pdf]](https://arxiv.org/pdf/2606.18661)</summary>

**Abstract:** Intelligent landslide hazard interpretation is critical for disaster prevention, yet current paradigms struggle to simultaneously extract visual features and high-level geoscientific semantics, while general-purpose vision-language models (VLMs) suffer from perceptual limitations and domain hallucinations in complex geological scenarios. To address these challenges, we propose an instruction-driven agentic framework comprising three components. First, LandslideBench, a multimodal fine-grained dataset with seven subtype labels, high-resolution imagery, pixel-level masks, and high-quality textual descriptions, is constructed via multi-VLM cross-validation and interactive annotation. Then, LandslideVLM, a landslide-oriented VLM, is fine-tuned via LoRA on LandslideBench to enhance geological semantic understanding. Finally, LandslideAgent, a domain rule-enhanced agent taking LandslideVLM as its cognitive backbone, employs a dual-rule controller incorporating structured report metadata constraints and cross-validation identification constraints to regulate automated tool invocation. Experiments demonstrate that LandslideBench provides effective baselines across five mainstream models on fine-grained classification and semantic segmentation. LandslideVLM achieves accuracy improvements of 10.96%, 32.87%, and 15.91% on landslide discrimination, fine-grained classification, and semantic description quality, respectively. LandslideAgent further enables autonomous multi-source spatial data inference, realizing full-process intelligence for landslide identification and analysis.

**arXiv ID:** 2606.18661
</details>

<details>
<summary><strong>SWE-Future: Forecast-Conditioned Data Synthesis for Future-Oriented Software Engineering Agents</strong> - Qiao Zhao, JianYing Qu, Jun Zhang, Yehua Yang, Hanwen Du, Zhongkai Sun - [[pdf]](https://arxiv.org/pdf/2606.18733)</summary>

**Abstract:** Realistic coding-agent benchmarks often replay public GitHub issues and pull requests, making them vulnerable to overlap with model pretraining, fine-tuning, synthetic-data generation, or benchmark-driven model selection. Fully synthetic tasks avoid direct historical replay, but can drift away from real repository needs. We propose SWE-Future, a forecast-conditioned data synthesis method for future-oriented coding tasks. Given a forecast snapshot at time $T_0$, the method uses only pre-$T_0$ repository evidence to forecast future feature implementation/enhancement, bugfix, and refactor task families. We first validate this forecasting step retrospectively: after forecasts are fixed, later pull requests are used only to measure whether the predicted task families match future repository work. In an 80-repository study, the forecaster achieves 58.1\% future-work relevance under the main semantic matching metric. We then use validated forecast families as conditioning signals to synthesize a 200-task coding-agent dataset across 61 repositories from a task-generation snapshot, rather than replaying the later pull requests used for validation. SWE-Future shows that repository-evolution forecasts can guide realistic, future-oriented coding-task synthesis while reducing direct dependence on historical pull-request replay.

**arXiv ID:** 2606.18733
</details>

<details>
<summary><strong>Data Intelligence Agents: Interpreting, Modeling, and Querying Enterprise Data via Autonomous Coding Agents</strong> - Anoushka Vyas, Aarushi Dhanuka, Sina Khoshfetrat Pakazad, Henrik Ohlsson - [[pdf]](https://arxiv.org/pdf/2606.19319)</summary>

**Abstract:** Production data integration is bottlenecked by repeated, lossy handoffs between data owners, engineers, and analysts who must collaboratively discover, structure, and query enterprise data. We present Data Intelligence Agents (DIA), a system of three agents (Data Interpreter, Schema Creator, and Query Generator) that compresses this workflow by treating autonomous coding agents (ACAs) as a first-class abstraction: rather than emitting text, the agents generate, execute, validate, and repair concrete artifacts, draw on a shared memory for experience reuse, and surface each for review by domain experts. DIA is deployed in production for enterprise customers. We study the Query Generator in depth and evaluate it in fully autonomous mode across seven SQL benchmarks spanning four task categories and four dialects. It matches or surpasses the best published results on all seven, demonstrating that an architecture grounded in execution, built on ACAs and a shared memory, generalizes across the data intelligence workload with adaptation confined to natural-language instructions.

**arXiv ID:** 2606.19319
</details>

<details>
<summary><strong>Notation Matters: A Benchmark Study of Token-Optimized Formats in Agentic AI Systems</strong> - Lorenz Kutschka, Bernhard Geiger - [[pdf]](https://arxiv.org/pdf/2605.29676)</summary>

**Abstract:** Large language models in Agentic AI systems consume tool schemas and execution results and emit tool invocations as structured data. The default language for that exchange, JSON, was designed for application-to-application interchange rather than token efficiency, so its structural elements impose substantial token overhead. Recent work proposes token-optimized alternatives such as TOON (Token-Oriented Object Notation) and TRON (Token Reduced Object Notation) as more compact replacements, but these formats have been evaluated only on isolated comprehension or generation tasks. Whether their token reductions hold inside end-to-end agentic loops therefore remains an open question. We evaluate TOON and TRON on four agentic benchmarks (BFCL, MCPToolBenchPP, MCP-Universe, StableToolBench) and five open-weight LLMs, decoupling input compression from output compression to measure comprehension and generation independently. TRON reduces tokens by up to 27% with accuracy within 14pp of the JSON baseline. TOON achieves up to 18% reduction at a similar 9pp accuracy cost, but additionally cascades on multi-turn parsing failures and collapses parallel tool-call output for most models. The code is available at: this https URL

**arXiv ID:** 2605.29676
</details>

<details>
<summary><strong>Dissecting model behavior through agent trajectories</strong> - Gaurav Gupta, Vatshank Chaturvedi, Jun Huan, Anoop Deoras - [[pdf]](https://arxiv.org/pdf/2606.17454)</summary>

**Abstract:** AI agent performance is not just a modeling problem, it is fundamentally a systems problem. The advanced capabilities of models are realized through agent harnesses. Therefore, a gap between model assumptions and harness behavior can easily prevent the model's full capabilities from translating into agent performance. We formalize this as the `intent-execution' gap: the mismatch between what the model intends and what the harness executes, and vice versa. We argue that minimizing this intent-execution gap is as important as other aspects of harness design such as tools and execution loops. To illustrate the impact of this harness-model alignment, we develop a simple and customizable harness called `Simple Strands Agent' (SSA). SSA aims to find the bulk of common patterns which generalize across different model families (such as Claude, Gemini, GPT, Grok, Qwen), as well as a small number of model-specific preferences. We make two contributions: (i) we reproduce or improve on the pass@1 performance reported by diverse model-provider families on popular agentic benchmarks (SWE-Pro, SWE-Verified and Terminal-Bench-2), and (ii) building on an analysis of 138k trajectories generated by SSA, we look beyond the pass@1 numbers which tend to be relatively even across frontier models. By representing agent trajectories in code state-spaces, we observe model-level differences in problem-solving behavior. Finer-grained metrics such as edit frequency, testing activity, and phase-transitions reveal how individual models allocate effort across different stages of autonomous problem solving.

**arXiv ID:** 2606.17454
</details>

<details>
<summary><strong>Your AI Travel Agent Would Book You a Bullfight: An Agentic Benchmark for Implicit Animal Welfare in Frontier AI Models</strong> - Jasmine Brazilek, Joel Christoph, Miles Tidmarsh, Carol Kline, Oliver Tullio, Arturs Kanepajs - [[pdf]](https://arxiv.org/pdf/2606.18142)</summary>

**Abstract:** AI agents are moving from advisors to actors, booking travel, planning menus, and running procurement on behalf of users. Existing benchmarks for AI and animal welfare evaluate model text responses to question-answer prompts, leaving open whether the welfare reasoning surfaced in those responses transfers to agentic deployment where the model must take actions with tools. We introduce TAC (Travel Agent Compassion), the first agentic benchmark measuring whether AI agents avoid options involving animal exploitation when acting on behalf of users. TAC presents an AI agent with twelve hand-authored travel booking scenarios across six categories of animal exploitation, augmented to forty-eight samples to control for price, rating, and position confounds. We evaluate seven frontier models from four labs. Every model scores below the chance level of sixty-four percent, with the best performer (Claude Opus 4.7) at fifty-three percent. A single welfare-aware sentence in the system prompt yields gains of forty-seven to sixty-three percentage points in Claude and GPT-5.5, twenty-six points in GPT-5.2, and under twelve points in DeepSeek and Gemini. An auxiliary Inspect Scout audit of 288 base-condition transcripts from the top two performers, using Gemini 2.5 Flash Lite as judge, flags zero transcripts for evaluation awareness, suggesting the below-chance rates do not stem from the models recognising the evaluation. We discuss implications for category-level variation across cultural domains, the limits of text-response welfare benchmarks, and the EU General-Purpose AI Code of Practice systemic risk framework.

**arXiv ID:** 2606.18142
</details>

<details>
<summary><strong>TWICE: Modeling the Temporal Evolution of Personalized User Behavior via Event-Driven Agents</strong> - Bingrui Jin, Kunyao Lan, Baihan LI, Mengyue Wu - [[pdf]](https://arxiv.org/pdf/2602.22222)</summary>

**Abstract:** User simulators are widely used for data generation, evaluation, and agent-based interaction, but existing approaches often model users as static personas or rely on generic historical context, making it difficult to capture how individual behavior evolves over time. To address this limitation, we propose TWICE, an LLM-based framework for temporally grounded personalized user simulation. TWICE combines structured user profiling, an event-driven memory module organized around life events and behavioral shifts, and a two-stage workflow separating event-grounded content planning from personalized style adaptation. This design enables the simulator to model not only what a user says, but also how past experiences shape later expression. We evaluate TWICE on a large-scale longitudinal Twitter dataset and introduce a comprehensive evaluation framework that jointly measures authenticity, consistency, and humanlikeness. Results show that TWICE consistently outperforms strong baselines, suggesting that event-centered memory is a promising mechanism for modeling the temporal evolution of personalized user behavior.

**arXiv ID:** 2602.22222
</details>

<details>
<summary><strong>VISUALSKILL: Multimodal Skills for Computer-Use Agents</strong> - Ziyan Jiang, Li An, Yujian Liu, Jiabao Ji, Qiucheng Wu, Jacob Andreas, Yang Zhang, Shiyu Chang - [[pdf]](https://arxiv.org/pdf/2606.18448)</summary>

**Abstract:** Computer-use agents (CUAs) approach human-level performance on standardised benchmarks but still struggle on long-horizon tasks and unseen software. Existing skill libraries address this with reusable skills, but represent the skill artifact as text only, despite the visual nature of GUI interaction. We propose VISUALSKILL: a hierarchical multimodal skill, tailored to each target application and organised as a central index over per-topic files, which the agent consumes through a load_topic MCP tool that fetches the relevant topic's text and figures on demand. We construct each skill with a two-stage pipeline that combines authored documentation with live-application UI exploration. On two CUA benchmarks, CUA-World and OSExpert-Eval, a Claude Code CLI agent backed by Claude Opus 4.6 reaches an average score of 0.456 with VISUALSKILL, a +15.3 point absolute lift over the no-skill baseline (0.303). Against a matched text-only skill that is generated from the same source content and differs from VISUALSKILL only in modality, VISUALSKILL yields a further +8.3 point absolute gain over the matched text-only skill (0.373 vs. 0.456), providing direct evidence that retaining visual figures in the skill artifact, rather than verbalizing them away, helps the agent both identify UI elements and verify workflow state after each action. Our code is available at this https URL.

**arXiv ID:** 2606.18448
</details>

<details>
<summary><strong>LegalWorld: A Life-Cycle Interactive Environment for Legal Agents</strong> - Songhan Zuo, Shengbin Yue, Tao Chiang, Guanying Li, Yun Song, Xuanjing Huang, Zhongyu Wei - [[pdf]](https://arxiv.org/pdf/2606.18728)</summary>

**Abstract:** Civil litigation is inherently a life-cycle process: what a lawyer drafts on day one constrains what unfolds at trial months later. Yet existing legal benchmarks evaluate isolated subtasks, and prior legal-agent simulators reinitialize each scenario from shared ground truth, leaving cross-stage causal dependencies unmodeled. We present LegalWorld, a life-cycle interactive environment that models Chinese civil litigation as a causally connected state chain of five stages (seven sub-scenarios), grounded in 75,309 paired Chinese civil judgments. We pair it with reusable infrastructure (local memory, global case memory, a Skill/Tool library) that keeps each dispute consistent across its full life cycle. Building on this environment, we construct LongJud-Bench to evaluate agent capability across all five connected stages. 18,992 ratings from 217 legal-background evaluators confirm that LegalWorld trajectories are procedurally faithful and role-consistent; and a capability-level cross-model evaluation reveals sharp divergences that aggregate scores cannot expose, with no single backbone leading across consultation, drafting, and courtroom advocacy. Detailed resources will be released publicly.

**arXiv ID:** 2606.18728
</details>

<details>
<summary><strong>HandwritingAgent: Language-Driven Handwriting Synthesis in Scalable Vector Space</strong> - Jaward Sesay, Yue Yu, Börje F. Karlsson - [[pdf]](https://arxiv.org/pdf/2606.18788)</summary>

**Abstract:** Teaching machines to emulate natural handwriting styles remains an open challenge, as it requires synthesizing stroke sequences that dynamically vary in shape, texture, pressure and script - not only across individuals, but also within a single person's handwriting. Attempts at this challenge have largely explored deep learning methods in both online and offline settings. However, these approaches are often constrained by style-specific architectural choices, heavy reliance on large datasets, high compute costs, and a lack of flexible control over writing styles through natural language. To this end, we introduce HandwritingAgent, a language-driven agent that can synthesize natural handwriting sequences directly in Scalable Vector Graphics (SVG) format with no need for style-specific training. The agent leverages a large reasoning model to geometrically analyse and autoregressively generate target handwritten glyphs as stroke sequences in a discrete grid canvas environment. Generation is conditioned on texts provided in either conversational or non-conversational mode, along with a reference handwriting-style image. Experiments on diverse handwriting tasks spanning imitation, recognition, multi-lingual handwriting synthesis, and generation of complex handwritten maths and science expressions indicate substantial improvement in performance, with HandwritingAgent matching or surpassing state-of-the-art generative handwriting models, while providing a more efficient, controllable, and generalizable synthesis method.

**arXiv ID:** 2606.18788
</details>

<details>
<summary><strong>LoHoSearch: Benchmarking Long-Horizon Search Agents Beyond the Human Difficulty Ceiling</strong> - Jiarui Zhao, Rongzhi Zhang, Lingchuan Liu, Hao Yang, Xunliang Cai, Xi Su - [[pdf]](https://arxiv.org/pdf/2606.12837)</summary>

**Abstract:** Search agent benchmarks exemplified by BrowseComp have rapidly saturated over the past year, with the strongest models surpassing 90% accuracy. Since these benchmarks are predominantly human-authored, annotators lack a global perspective on entity statistics and cannot systematically maximize search space size and structural complexity. This creates a difficulty ceiling that is hard to break. To address this, we introduce LoHoSearch (Long-Horizon Search Agents), a challenging benchmark comprising 544 human-verified questions across 11 domains. LoHoSearch is constructed via an automated pipeline built upon a knowledge graph covering over 7 million Wikipedia entities, which selects relations with large search spaces and assembles them into structurally complex questions with KG-verified unique answers. Our evaluation demonstrates that even the strongest model achieves only 34.74% accuracy, and existing context management strategies (best +6.8%) yield far smaller gains than on prior benchmarks. LoHoSearch provides a more demanding standard for evaluating long-horizon reasoning and context management in search agents.

**arXiv ID:** 2606.12837
</details>

<details>
<summary><strong>GRACE-DS: a Guarded Reward-guided Agent Correction Environment in Data Science</strong> - Aleksandr Tsymbalov, Danis Zaripov, Artem Epifanov, Anastasiya Palienko - [[pdf]](https://arxiv.org/pdf/2606.16000)</summary>

**Abstract:** We introduce GRACE-DS, a Guarded Reward-guided Agent Correction Environment in Data Science for pre-deployment evaluation of LLM-powered AutoML agents. GRACE-DS is a set of evaluation metrics in an isolated environment that can be applied to tabular ML tasks specific to a particular organization. It exposes agents to realistic workflow stages, from planning and data inspection through feature engineering, model development, validation, and code repair to final submission, while hidden executable validators measure not only final predictive performance but also leakage avoidance, reproducibility, protocol validity, correction behavior, and reward alignment. The strongest structured regime, flexible iterative interaction (our approach), achieves higher end-to-end normalized hidden-test quality than single-shot generation, unstructured interaction, and restart-based baselines, while also improving protocol-valid completion. Validated across more than 7,000 episodes, these results establish GRACE-DS as a robust platform for assessing the capacity of LLM-based AutoML agents to execute machine learning workflows under production-like conditions and in accordance with organization-specific requirements.

**arXiv ID:** 2606.16000
</details>

<details>
<summary><strong>ROBOSHACKLES: A Safety Dataset for Human-Injury Prevention in Embodied Foundation Models</strong> - Zhuowen Yin, Chongyang Liu, Wenzhang Yang, Renjue Li, Yinxing Xue - [[pdf]](https://arxiv.org/pdf/2606.18632)</summary>

**Abstract:** Embodied Foundation Models (EFMs) integrate multimodal understanding, future-state reasoning, and executable robot actions. Yet their safety alignment for human-injury prevention remains underexplored, primarily because real-world data of robots harming humans or creating hazardous household situations cannot be safely or ethically collected. To address this challenge, we propose a safety-critical data construction pipeline for human-injury prevention in this http URL from real DROID observations, our construction pipeline proceeds through scene understanding, hazard-aware image editing, temporal prompt generation, and single-pass rollout synthesis. The temporal prompts specify the expected scene evolution, while Wan2.7 synthesizes realistic robotic rollouts from the edited hazardous states in a single pass. Using this pipeline, we construct ROBOSHACKLES, a 10,000-clip robotic video dataset derived from real DROID observations, spanning two direct-harm and four indirect-harm categories. To ensure dataset quality, we assess task completion and visual quality with automatic metrics, and evaluate six representative EFMs under a refusal-based safety criterion. Results show that all evaluated models produce unsafe actions in the tested safety-critical scenarios, yielding a 100% unsafe action generation rate. ROBOSHACKLES serves as a scalable benchmark and training resource for refusal learning and hazard anticipation before robot action this http URL dataset is publicly available at this https URL.

**arXiv ID:** 2606.18632
</details>

<details>
<summary><strong>ERQA-Plus: A Diagnostic Benchmark for Reasoning in Embodied AI</strong> - Hong Yang, Basura Fernando - [[pdf]](https://arxiv.org/pdf/2606.17639)</summary>

**Abstract:** Generalist embodied agents require more than object recognition: they must reason about spatial relations, actions, procedures, human intentions, environmental constraints, and commonsense consequences from situated visual observations. Yet existing visual and embodied question answering benchmarks often provide limited control over the reasoning dependencies being tested, making it difficult to distinguish grounded embodied reasoning from shortcut-driven visual or linguistic pattern matching. We present ERQA-Plus, a diagnostic benchmark for reasoning in embodied AI. ERQA-Plus contains 1,766 question-answer instances grounded in 711 robot-centric images and organized according to a structured taxonomy spanning perceptual, action-centric, social-interaction, navigation-environmental, and contextual commonsense reasoning. The dataset is constructed using a multi-stage generation and validation pipeline that combines taxonomy-guided question generation, automatic quality judging, iterative revision, and human assessment to improve visual grounding, answer validity, and reasoning quality. We benchmark representative general-purpose vision-language models and embodied models, including LLaVA-NeXT-8B, Prismatic-7B, MiniCPM-V-4.5-8B, Qwen3-VL, RoboRefer-8B, and RoboBrain2.5-8B. Although the strongest model, Qwen3-VL-32B, achieves 83.4% overall accuracy and 61.4 SBERT score, category-level results reveal persistent weaknesses in spatial reasoning, procedural reasoning, event prediction, and intention inference. ERQA-Plus therefore provides a fine-grained evaluation framework for measuring not only whether embodied agents answer correctly, but also which forms of embodied reasoning they can and cannot perform reliably. The dataset is available this https URL and the project page at this https URL.

**arXiv ID:** 2606.17639
</details>

<details>
<summary><strong>HANSEL: Extracting Breadcrumbs from Web Agent Trajectories for Interactive Verification</strong> - Yujin Zhang, Daye Nam - [[pdf]](https://arxiv.org/pdf/2606.18671)</summary>

**Abstract:** AI web agents can perform complex, multi-step tasks such as searching for products, comparing options, and making purchases on behalf of users. However, verifying the correctness of an agent's output remains difficult. Existing transparency mechanisms, including full trajectory logs, source links, screenshots, and LLM-generated summaries, treat verification as a passive reading task, leaving users to sift through overwhelming logs or trust potentially unfaithful explanations. We present HANSEL (Highlighting Agent Navigation Steps as Evidence Links), a system that extracts interactive, verifiable evidence from web-agent trajectories. Given an agent trajectory, HANSEL extracts evidence pages and snippets and presents them as navigable, interactive views with relevant page state preserved (e.g., applied filters, search queries, and scroll positions), enabling users to verify how the agent arrived at its answer. When the agent's answer cannot be traced to any visited page, HANSEL explicitly flags this gap. A technical evaluation on 45 tasks from AssistantBench and Online-Mind2Web shows that HANSEL achieves 83.7% precision and 88.8% recall in identifying evidence pages, while reducing trajectory volume by 61.6%. In a controlled user study with 14 participants, HANSEL significantly reduced task completion time and perceived effort compared to a standard agent interface, while participants rated it significantly higher on usability, verification ease, and error identification. Our results demonstrate that reframing verification as an interactive activity, rather than passive consumption of agent explanations, leads to more efficient human oversight of AI agents.

**arXiv ID:** 2606.18671
</details>

</details>

<details open>
<summary><h2>LLM Agents (14 papers)</h2></summary>

<details>
<summary><strong>CEO-Bench: Can Agents Play the Long Game?</strong> - Haozhe Chen, Karthik Narasimhan, Zhuang Liu - [[pdf]](https://arxiv.org/pdf/2606.18543)</summary>

**Abstract:** Language model agents are becoming proficient executors at isolated, short-horizon tasks such as software engineering and customer service. Yet real-world challenges require a combination of sophisticated skills that remain largely untested in agents: (1) navigating long horizons amid uncertainty; (2) acquiring information in noisy environments; (3) adapting to a changing world; (4) orchestrating multiple moving parts toward a coherent goal. We introduce CEO-Bench, which evaluates these capabilities together by simulating a representative real-world task: operating a startup for 500 days. An agent manages pricing, marketing, budgeting, and many other aspects of a fictional company through a programmable Python interface, operating in the same environment and facing the same challenges as a human CEO. Success demands analyzing noisy, interconnected business databases, translating signals into sound strategy, and coordinating many decisions with programming. The strongest agents write sophisticated code that simulates customer cohorts to forecast future cash and mines negotiation history to uncover hidden customer preferences. Even so, most state-of-the-art models struggle in this environment. Only Claude Opus 4.8 and GPT-5.5 finish above the $1M starting balance, and neither consistently turns a profit. CEO-Bench takes a first step toward measuring the intelligence required to drive sustained, adaptive progress over time.

**arXiv ID:** 2606.18543
</details>

<details>
<summary><strong>ProfiLLM: Utility-Aligned Agentic User Profiling for Industrial Ride-Hailing Dispatch</strong> - Tengfei Lyu, Zirui Yuan, Xu Liu, Kai Wan, Zihao Lu, Li Ma, Hao Liu - [[pdf]](https://arxiv.org/pdf/2606.18803)</summary>

**Abstract:** Bringing Large Language Models (LLMs) into industrial ride-hailing dispatch as semantic feature extractors over platform-scale behavioral logs is a compelling but under-explored data systems problem. Production matching pipelines remain dominated by structured numerical features, yet decisive behavioral signals (e.g., a driver's habitual aversion to certain regions) are inherently contextual and naturally expressible as LLM-generated user profiles. However, scaling such profiling to a live, millisecond-latency dispatcher faces three intertwined constraints rarely addressed together: on a platform with millions of daily orders, logs exceed any LLM's context window by orders of magnitude; most users are long-tail, with too few interactions for per-user profiling; and surface-fluent profiles do not necessarily improve downstream prediction utility. We present ProfiLLM, an agentic LLM data pipeline that operationalizes utility-aligned user profiling for production matching systems through two modules. (1) Tool-Augmented Global Knowledge Mining equips an LLM agent with 27 analytical tools to mine platform-scale data, producing reusable global knowledge, adaptive user clustering rules, and region-level supply-demand priors. (2) Utility-Aligned Profile Exploration generates multiple candidate profiles per cluster, evaluates them via a lightweight downstream utility proxy, iteratively refines the best candidates and constructs preference pairs for DPO fine-tuning. Deployed on DiDi's production dispatcher, ProfiLLM achieves up to +6.14% relative AUC improvement in outcome prediction, up to +4.35% GMV gain in dispatching simulation, and consistent improvements in a 14-day online A/B test including +0.47% GMV, +0.33% Completion Rate, and -0.82% Cancel-Before-Accept rate.

**arXiv ID:** 2606.18803
</details>

<details>
<summary><strong>Decoupling Search from Reasoning: A Vendor-Agnostic Grounding Architecture for LLM Agents</strong> - Emmanuel Aboah Boateng, Kyle MacDonald, Amardeep Kumar, Siddharth Kodwani, Sudeep Das - [[pdf]](https://arxiv.org/pdf/2606.18947)</summary>

**Abstract:** Production LLM agents increasingly depend on real-time search, yet native search grounding bundles retrieval policy, provider choice, evidence injection, cost, latency, and generation behavior behind a single model-provider boundary. This coupling makes grounding hard to inspect, tune, reuse, or port, and can trigger Search-Induced Verbosity that breaks strict output contracts. We present Decoupled Search Grounding (DSG), a vendor-agnostic boundary that moves grounding outside the reasoning model through an MCP-compatible gateway, exposing provider routing, source-aware context rendering, configured fallback, retrieval-depth control, and exact plus semantic caching as first-class controls. Across five frontier models on SimpleQA, FreshQA, and HotpotQA, native search leads on recency-sensitive FreshQA, but DSG exposes a stronger frontier when control matters: on SimpleQA it nearly matches native accuracy (86.1% vs. 87.7%) at 91% lower search cost, preserves concise answer contracts, and reaches a 99.4% warm-cache hit rate with 68% lower latency. Deployed as a shared production grounding layer for large-scale agentic workloads with interchangeable models, DSG matches or slightly exceeds native-search accuracy on an e-commerce query-understanding (QIU) workload while cutting search cost by over 98%. Real-time grounding is best treated as an optimizable interface boundary, not a fixed model feature.

**arXiv ID:** 2606.18947
</details>

<details>
<summary><strong>SafeClawBench: Separating Semantic, Audit-Evidence, and Sandbox Harm in Tool-Using LLM Agents</strong> - Yuchuan Tian, Mengyu Zheng, Haocheng Mei, Ye Yuan, Chao Xu, Xinghao Chen, Hanting Chen, Yu Wang - [[pdf]](https://arxiv.org/pdf/2606.18356)</summary>

**Abstract:** Tool-using language-model agents introduce security failures that go beyond unsafe text: they can disclose protected objects, write persistent memory, send messages, modify databases, or trigger harmful code and tool effects. Existing evaluations often collapse these stages into a single attack success rate, making it difficult to tell whether a model merely agreed with an attacker or actually produced observable harm. We introduce SafeClawBench, a staged benchmark for tool-using agent security with 600 controlled adversarial tasks across six attack families: direct and indirect prompt injection, tool-return injection, memory poisoning, memory extraction, and ambiguity-driven unsafe inference. SafeClawBench reports three separate endpoints: semantic attack acceptance, audit-visible harm evidence, and sandbox-observed tool/state harm. Evaluating five agent endpoints under four prompt-level policies, we find that these endpoints capture different failure modes. Without additional prompt protection, semantic failure rates vary widely across models, from 9.0% to 44.2%. Audited harm evidence is narrower than semantic failure, and under a separate executable protocol some matched task identities produce sandbox harm despite passing the Semantic Core call: in a 12,000-row matched analysis, 291 of 347 observed sandbox harms occur in rows that pass the semantic check. Prompt policies change endpoint outcomes, but their effects depend on both model and protocol. SafeClawBench provides a reproducible framework for comparing agent models and prompt-policy conditions without conflating textual compliance, evidence-supported harm, and executable state changes. The open-source dataset is available at this https URL.

**arXiv ID:** 2606.18356
</details>

<details>
<summary><strong>LLMZero: Discovering Adaptive Training Strategies for RL Post-Training via LLM Agents</strong> - Haoyang Fang, Wei Zhu, Boran Han, Alex Zhang, Zhenyu Pan, Shuo Yang, Shuai Zhang, Jiading Gai, Peng Tang, Cuixiong Hu, Xuan Zhu, Huzefa Rangwala, George Karypis, Bernie Wang - [[pdf]](https://arxiv.org/pdf/2606.18388)</summary>

**Abstract:** RL post-training strategies are dataset-dependent and reveal a recurring empirical pattern: capacity parameters accumulate monotonically across stages, while regularization parameters predominantly oscillate in response to shifting training dynamics. This distinction matters because fixed schedules commit all parameters to fixed trajectories and therefore cannot express the non-stationary exploration-exploitation tradeoffs that regularization must track; the principle provides actionable design rules for multi-stage training. We discover this through LLMZero, a system where LLM agents search over training trajectories via tree search, diagnosing pathologies at each checkpoint and proposing coordinated multi-parameter transitions. Across 4 diverse GRPO tasks, LLMZero discovers strategies that improve over the base model by 9% to 140% relative and over grid search by 6% to 15% relative, consistently outperforming random search and the skill-based agent. The structural principle transfers across tasks, providing an explanation for why discovered strategies take qualitatively different forms yet share similar parameter dynamics.

**arXiv ID:** 2606.18388
</details>

<details>
<summary><strong>Code-Augur: Agentic Vulnerability Detection via Specification Inference</strong> - Zhengxiong Luo, Mehtab Zafar, Dylan Wolff, Abhik Roychoudhury - [[pdf]](https://arxiv.org/pdf/2606.18619)</summary>

**Abstract:** The advent of agentic vulnerability detection is already becoming a watershed moment for software security. Audits conducted entirely by autonomous LLM agents are uncovering critical vulnerabilities in fundamental software underpinning digital society. Many of these vulnerabilities remained masked for years, surfacing only now with AI agents. Yet the reasoning behind these discoveries remains alarmingly opaque and unvalidated. What assumptions did the agent make about a function's inputs when it deemed that function to be secure? Failures in reasoning and incorrect assumptions can lead to missed vulnerabilities and reduce trust in agentic analysis.
We propose a security-specification-first paradigm that (1) exposes the agent's tacit assumptions explicitly as security specifications and (2) continuously refines those specifications via runtime falsification. We realize our approach in Code-Augur, a novel harness for agentic vulnerability detection. Given a codebase, Code-Augur analyzes each component of the system for vulnerable code. When it deems a component to be secure, it commits the local invariants behind that judgment as in-source assertions. In parallel, Code-Augur leverages a guided fuzzer to attempt to falsify those assumptions. When the fuzzer triggers an assertion, this either reveals a genuine vulnerability or a flawed specification to refine. In both cases, this process grounds the agent's understanding, aligning its view of code intent with how the code actually behaves. On real-world subjects, Code-Augur effectively leverages security specifications to detect more vulnerabilities than other state-of-the-art agents. Additionally, Code-Augur found 22 new vulnerabilities in key open-source projects. Compared to curated specialized models like Claude Mythos, Code-Augur offers effective agentic vulnerability detection built on widely available LLMs like Sonnet and DeepSeek.

**arXiv ID:** 2606.18619
</details>

<details>
<summary><strong>Structured Cognitive Loop for Behavioral Intelligence in Large Language Model Agents (Extended Revision: From Behavioral Architecture to Epistemic Accountability)</strong> - Myung Ho Kim - [[pdf]](https://arxiv.org/pdf/2510.05107)</summary>

**Abstract:** The central challenge for AI agents is not only performance but accountability. Agents that act through opaque prompt sequences may produce correct outputs, but they provide little basis for verifying why an action was permitted, where an error occurred, or how responsibility should be assigned. This paper presents the Structured Cognitive Loop as an architecture for accountable behavior in large language model agents. SCL separates cognition, memory, control, and action into distinct modules. The language model proposes. External memory preserves verified state. A lightweight controller checks preconditions, prevents redundant actions, and authorizes execution before tools are used. We evaluate SCL against ReAct and common LangChain agent variants across travel planning, conditional email drafting, and constraint guided image generation. Across 360 episodes, SCL achieves 86.3 percent task success compared with 70.5 to 76.8 percent for prompt based baselines. It also improves goal fidelity, reduces redundant tool calls, increases reuse of intermediate state, and lowers unsupported assertions. This extended revision situates SCL within a broader architecture of epistemic accountability. Subsequent extensions integrate context aware Human in the Loop control, Pool Gated Retrieval, and the Horizon Warrant Commitment framework. Together these components define an agent architecture in which the model proposes, structure decides, evidence is warranted before use, and human judgment is embedded in the trace rather than imposed after the fact. The result is a foundation for AI agents whose decisions are not only effective but also authorized, inspectable, and accountable.

**arXiv ID:** 2510.05107
</details>

<details>
<summary><strong>InfoPO: Information-Driven Policy Optimization for User-Centric Agents</strong> - Fanqi Kong, Jiayi Zhang, Mingyi Deng, Chenglin Wu, Yuyu Luo, Bang Liu - [[pdf]](https://arxiv.org/pdf/2603.00656)</summary>

**Abstract:** Real-world user requests to LLM agents are often underspecified. Agents must interact to acquire missing information and make correct downstream decisions. However, current multi-turn GRPO-based methods often rely on trajectory-level reward computation, which leads to credit assignment problems and insufficient advantage signals within rollout groups. A feasible approach is to identify valuable interaction turns at a fine granularity to drive more targeted learning. To address this, we introduce InfoPO (Information-Driven Policy Optimization), which frames multi-turn interaction as a process of active uncertainty reduction and computes an information-gain reward that credits turns whose feedback measurably changes the agent's subsequent action distribution compared to a masked-feedback counterfactual. It then combines this signal with task outcomes via an adaptive variance-gated fusion to identify information importance while maintaining task-oriented goal direction. Across diverse tasks, including intent clarification, collaborative coding, and tool-augmented decision making, InfoPO consistently outperforms prompting and multi-turn RL baselines. It also demonstrates robustness under user simulator shifts and generalizes effectively to environment-interactive tasks. Overall, InfoPO provides a principled and scalable mechanism for optimizing complex agent-user collaboration. Code is available at this https URL.

**arXiv ID:** 2603.00656
</details>

<details>
<summary><strong>SkillRevise: Improving LLM-Authored Agent Skills via Trace-Conditioned Skill Revision</strong> - Yuxuan Liu, Zhaochen Su, Lingyun Xie, Yuhao Zhang, Qing Zong, Jiahe Guo, Zhongwei Xie, Yiyan Ji, Yauwai Yim, Hongyu Luo, Xiyu Ren, Ruan Chenyu, Haoran Li, Yangqiu Song - [[pdf]](https://arxiv.org/pdf/2606.01139)</summary>

**Abstract:** Agent skills are procedural artifacts that enable LLM agents to execute workflows, verify constraints, and recover from failures. Existing self-evolving methods refine skills using accumulated trajectories. However, they struggle in cold-start settings, where only an initial, imperfect skill is available. Consequently, skill construction defaults to expert authoring or one-shot LLM generation. Expert-authored skills are costly and may not align with how LLM agents actually execute tasks, while one-shot generated skills can be syntactically well formed yet behaviorally weak. To bridge this gap, we propose SkillRevise, an execution-grounded framework designed to iteratively refine these initial skills. SkillRevise diagnoses skill defects from execution evidence, retrieves relevant repair principles from a general memory, and applies execution-anchored edits. By re-executing candidates, it retains the first verifier-passing skill within the revision budget and falls back to empirical utility only when no candidate succeeds. Evaluated across three benchmarks and five LLMs, SkillRevise substantially outperforms one-shot baselines, improving the base agent's success rate on SkillsBench from 36.05% to 61.63%. Furthermore, the revised skills transfer across both executors and task environments, suggesting that SkillRevise captures reusable procedural knowledge beyond any single executor.

**arXiv ID:** 2606.01139
</details>

<details>
<summary><strong>MapSatisfyBench: Benchmarking Satisfaction-Aware Map Agents through Behavior-Grounded Implicit Decision Factors</strong> - Lubin Bai, Mengyu Cao, Sixue Wang, Zhongwei Wan, Yue Pan, Jiale Hou, Xiang Li, Xiuyuan Zhang - [[pdf]](https://arxiv.org/pdf/2606.17453)</summary>

**Abstract:** Large language model agents are increasingly integrated into map services. Since map services are embedded in everyday-life scenarios rather than professional task settings, users often express their needs informally, resulting in underspecified queries with many unspoken needs, namely, implicit decision factors that are critical for user satisfaction. Although clarification is an effective way to mitigate this issue, it increases user burden in daily interaction, and a capable agent should first proactively recover such factors from available information sources. However, evaluating this ability is challenging. The first challenge is to determine which implicit decision factors are suitable for evaluation. A factor is evaluable only if it affects user acceptance and can be recovered from information available to the agent before it responds. Second, user satisfaction cannot be reliably represented by a single reference answer, requiring a benchmark that converts satisfaction-relevant factors into objective and quantifiable evaluation targets. To address these challenges, we propose a restore-identify-filter framework that reconstructs complete user needs from behavior-chain evidence, identifies implicit decision factors, and retains only those supported by pre-query evidence. Building on this methodology, we construct MapSatisfyBench from large-scale, real-world anonymized user data and annotate ground truth from five dimensions and enables full-chain evaluation of satisfaction-aware map agents. Experiments show that current agents generally perform well on explicit task completion, but remain limited in satisfying implicit decision factors and proactively acquiring the evidence needed for satisfaction-aware decisions. These findings establish MapSatisfyBench as a benchmark for shifting map-agent evaluation from task completion toward satisfaction-aware spatial decision making.

**arXiv ID:** 2606.17453
</details>

<details>
<summary><strong>ActMem: Bridging the Gap Between Memory Retrieval and Reasoning in LLM Agents</strong> - Xiaohui Zhang, Zequn Sun, Chengyuan Yang, Yaqin Jin, Yazhong Zhang, Wei Hu - [[pdf]](https://arxiv.org/pdf/2603.00026)</summary>

**Abstract:** Memory management is essential for LLM agents in long-term interactions. Current memory frameworks typically treat agents as passive ``recorders'' and retrieve information without understanding its deeper implications. They may fail in scenarios requiring reasoning and complex decision-making. To bridge this critical gap, we propose a novel actionable memory framework called ActMem that integrates memory retrieval with active causal reasoning. ActMem transforms unstructured dialogue history into a structured causal and semantic graph. By leveraging counterfactual reasoning and commonsense completion, it enables agents to deduce implicit constraints and resolve potential conflicts between past states and current intentions. Furthermore, we introduce a comprehensive dataset ActMemEval to evaluate agent reasoning capabilities in logic-driven scenarios, moving beyond the fact-retrieval focus of existing memory benchmarks. Experiments demonstrate that ActMem significantly outperforms baselines in handling complex, memory-dependent tasks, paving the way for more consistent and reliable intelligent assistants.

**arXiv ID:** 2603.00026
</details>

<details>
<summary><strong>GateMem: Benchmarking Memory Governance in Multi-Principal Shared-Memory Agents</strong> - Zhe Ren, Yibo Yang, Yimeng Chen, Zijun Zhao, Benshuo Fu, Zhihao Shu, Bingjie Zhang, Yangyang Xu, Dandan Guo, Shuicheng Yan - [[pdf]](https://arxiv.org/pdf/2606.18829)</summary>

**Abstract:** Memory benchmarks for LLM agents largely assume single-user settings, leaving shared assistants for hospitals, workplaces, campuses, and households understudied. In these deployments, multiple principals write to a common memory pool and query it under different roles, scopes, and relationships, so memory quality requires governance as well as recall. We introduce GateMem, a benchmark for multi-principal shared-memory agents. GateMem jointly evaluates utility for legitimate long-horizon requests with state updates, access control across contextual authorization boundaries, and agent-facing active forgetting after explicit deletion requests. It spans medical, office, education, and household domains, with long-form multi-party episodes, incremental memory injection, hidden checkpoints, structured judging, and leak-target annotations. Across diverse baselines and backbone models, no method simultaneously achieves strong utility, robust access control, and reliable forgetting. Long-context prompting often yields the best governance score at high token cost, while retrieval-based and external-memory methods reduce cost yet still leak unauthorized or deleted information. These results show current memory agents remain far from reliable shared institutional deployment.

**arXiv ID:** 2606.18829
</details>

<details>
<summary><strong>EvoArena: Tracking Memory Evolution for Robust LLM Agents in Dynamic Environments</strong> - Jundong Xu, Qingchuan Li, Jiaying Wu, Yihuai Lan, Shuyue Stella Li, Huichi Zhou, Bowen Jiang, Lei Wang, Jun Wang, Anh Tuan Luu, Caiming Xiong, Hae Won Park, Bryan Hooi, Zhiyuan Hu - [[pdf]](https://arxiv.org/pdf/2606.13681)</summary>

**Abstract:** Large language model (LLM) agents have achieved strong performance on a wide range of benchmarks, yet most evaluations assume static environments. In contrast, real-world deployment is inherently dynamic, requiring agents to continually align their knowledge, skills, and behavior with changing environments and updated task conditions. To address this gap, we introduce EvoArena, a benchmark suite that models environment changes as sequences of progressive updates across terminal, software, and social domains. We further propose EvoMem, a patch-based memory paradigm that records memory evolution as structured update histories, enabling agents to reason about environmental evolution through changes in their memory. Experiments show that current agents struggle on EvoArena, achieving an average accuracy of 39.6% across evolving terminal, software, and social-preference domains. EvoMem consistently improves performance, yielding an average gain of 1.5% on EvoArena and also improving standard benchmarks such as GAIA and LoCoMo by 6.1% and 4.8%. Beyond individual tasks, EvoMem further improves chain-level accuracy by 3.7% on EvoArena, where success requires completing a consecutive sequence of related evolutionary subtasks. Mechanistic analysis shows that EvoMem improves evidence capture in the memory, indicating better preservation of complete evolving environment states. Our results highlight the importance of modeling evolution in both evaluation and memory for reliable agent deployment.

**arXiv ID:** 2606.13681
</details>

<details>
<summary><strong>GrowthHacker: Automated Off-Policy Evaluation Optimization Using Code-Modifying LLM Agents</strong> - Jie JW Wu, Ayanda Patrick Herlihy, Ahmad Saleem Mirza, Ali Afoud, Fatemeh Fard - [[pdf]](https://arxiv.org/pdf/2511.00802)</summary>

**Abstract:** With data-driven development now widely adopted, online A/B testing is an established method for measuring the effects of new technologies. However, deploying online experiments demands resources for design, implementation, and deployment, and may negatively impact users (e.g., unsafe or unethical outcomes) while requiring weeks of data collection. To address this, the growing research area of off-policy evaluation (OPE), or offline A/B testing, assesses new technologies offline using previously collected logged data. OPE is also a fundamental problem in reinforcement learning and is important where online testing is expensive or risky, such as healthcare, recommender systems, education, and robotics. Despite advances in code-generation large language models (LLMs) and agentic workflows, little is known about whether and how LLMs and LLM-based agents can automatically optimize OPE implementations. We propose GrowthHacker, a benchmark that evaluates baseline LLMs and LLM-based agents on large-scale public datasets. GrowthHacker autonomously and iteratively modifies code, runs OPE, and uses the metrics to guide subsequent optimization. We evaluate methods on Open Bandit Pipeline (OBP) and Scope-RL, and develop a two_agent framework that addresses limitations of existing frameworks while reducing complexity. Across both libraries, two_agent shows the highest reliability (98.1%-100% success rate) and positive-outcome rate (78%), with a median improvement of 4.4% among positive outcomes; CrewAI achieves the highest average improvement (37.9%) and is the only framework with zero extreme-value failures. AutoGen and Default each reach 65% positive-outcome rates. These results establish the feasibility of using LLM-based agents as automated "growth hackers" to continuously improve OPE systems, with implications for scaling data-driven decision-making where manual optimization is expensive.

**arXiv ID:** 2511.00802
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (27 papers)</h2></summary>

<details>
<summary><strong>R2D-RL: A RoboCup 2D Soccer Environment for Multi-Agent Reinforcement Learning</strong> - Haobin Qin, Baofeng Zhang, Hidehisa Akiyama, Keisuke Fujii - [[pdf]](https://arxiv.org/pdf/2606.18786)</summary>

**Abstract:** Robot soccer is a challenging testbed for multi-agent reinforcement learning because it combines partial observability, cooperative and adversarial interaction, sparse rewards, and long-horizon tactical behavior. RoboCup 2D Soccer Simulation (RCSS2D) provides a mature robot-soccer platform, but its competition-oriented server-client architecture is difficult to use directly with modern Python-based MARL workflows. We introduce R2D-RL, a reinforcement learning environment that connects RCSS2D and HELIOS-based player clients to a Python MARL interface through shared-memory communication and cycle-level synchronization. R2D-RL supports full-field and scenario-based training with configurable opponents, Base discrete and Hybrid parameterized action spaces, action masks, expected possession value (EPV)-based reward shaping, and parallel execution. We provide front-goal scenarios and an 11-vs-11 full-field benchmark, together with baseline results.

**arXiv ID:** 2606.18786
</details>

<details>
<summary><strong>Simulating Hate Speech Cascades with Multi-LLM Agents: Empirical Grounding, Modeling Fidelity, and Intervention Strategies</strong> - Fan Huang - [[pdf]](https://arxiv.org/pdf/2606.18264)</summary>

**Abstract:** Faithful modeling of hateful content propagation on online platforms remains an open problem for moderation research. Classical cascade models that do not explicitly represent the profile, community, and content factors associated with hateful-content propagation may yield moderation strategies that behave less effectively when deployed in real-world scenarios. Multi-agent large language model (LLM) systems can, in principle, make each reshare decision depend on the user's profile, the surrounding community, and the post's content, but it remains unclear whether this added flexibility actually reproduces real hateful cascades more faithfully than classical baselines. We study three hateful Bluesky cascades and a size-matched benign control. In the empirical Bluesky data, we found that: 97.4--99.7\% of reposters take a hostile stance; toxicity-engagement homophily is higher on the diffusion tree than on the follower graph for hateful cascades; topology is star-like for the hateful cascades (most reposts come directly from the root) versus tree-like for the benign cascade (reposts propagate through multi-hop chains). In simulation, a multi-LLM-agent simulator reproduces the stance monoculture and the toxicity-delta direction. A structured ablation identifies agent heterogeneity as the leading fidelity factor, and amplifier targeting on dense networks yields 7.5--12.9\% reduction at 5.7\% benign collateral.

**arXiv ID:** 2606.18264
</details>

<details>
<summary><strong>Towards Multi-Agent-Simulation-Based Community Note Evaluation</strong> - Changxi Wen, Shuning Zhang, Bohao Chu, Yuwei Chuai, Hui Wang, Dai Shi, Xin Yi, Hewu Li - [[pdf]](https://arxiv.org/pdf/2606.18268)</summary>

**Abstract:** Community-based fact-checking that relies on cross-consensus is expanding rapidly on social media platforms. However, the delay and low-ratio of cross-consensus community fact-checks rated by human contributors remains a significant challenge. To address this, we first created ComRate, a large-scale dataset comprising 2.5 million community notes and over 209 million ratings sourced from $\mathbb{X}$. We then propose MultiCom, a persona-guided multi-agent rating framework for community note evaluation. MultiCom simulates diverse rater population by clustering contributors in a matrix-factorized rater space and prompting persona agents to generate structured assessments based on the official community notes rating schema. These agents output structured and explainable judgments, such as confidence, agreement signals and reasons. An out-of-fold calibrated aggregation algorithm combines features such as raw votes and diagnostic reason signals for reliable prediction. Extensive evaluations demonstrate that MultiCom outperforms alternative methods, achieving an average accuracy of 84.7% (balanced accuracy 68.3%, macro-F1 60.1%) on the evaluation set.

**arXiv ID:** 2606.18268
</details>

<details>
<summary><strong>Mitigating Anchoring Bias in LLM-Based Agents for Energy-Efficient 6G Autonomous Networks</strong> - Hatim Chergui, Claudia Carballo González, Farhad Rezazadeh, Merouane Debbah - [[pdf]](https://arxiv.org/pdf/2606.18272)</summary>

**Abstract:** This paper presents an autonomous agentic resource negotiation framework designed to enable zero-touch network slicing in 6G architectures using Large Language Model (LLM) agents. While LLMs offer powerful reasoning capabilities, we demonstrate that such agents inherently suffer from anchoring bias, rigidly adhering to initial heuristic proposals and causing severe network over-provisioning. To systematically mitigate this cognitive bias, we propose a novel randomized anchoring strategy modeled via a Truncated 3-Parameter Weibull distribution. This mathematically bounded approach seamlessly integrates with burst-aware Digital Twins (DTs) employing Conditional Value at Risk (CVaR) to rigorously guarantee strict Service Level Agreement (SLA) tail-latencies. To validate our methodology, we introduce and prove the \emph{Bimodal Constraint-Avoidance Utility Theorem}, demonstrating that while feasible negotiations follow classical convex bounds, highly constrained scenarios undergo a phase transition governed by an inverse rational decay envelope. Empirical results generated using a locally hosted 1B-parameter model (\texttt{otel-llm-1b-it}) confirm these dual-regime bounds. Our cognitive de-biasing successfully dismantles rigid negotiation patterns, forcing agents into active exploration to safely ride SLA boundaries and boost system energy savings up to 25\%. Crucially, the lightweight 1B LLM achieves sub-second inference latencies (0.95s mean), ensuring our multi-agent framework is compatible with the operational timescales of the O-RAN non-Real-Time RAN Intelligent Controller (non-RT RIC)\footnote{Our source code is available for non-commercial use at this https URL.

**arXiv ID:** 2606.18272
</details>

<details>
<summary><strong>TRIDENT: Breaking the Hybrid-Safety-Physics Coupling for Provably Safe Multi-Agent Reinforcement Learning</strong> - Zijie Meng, Ziwei Li, Yufei Liu, Zhiyu Li, Jiyuan Liu, Wenhua Nie, Bingcai Wei, Miao Zhang - [[pdf]](https://arxiv.org/pdf/2606.18308)</summary>

**Abstract:** Safe coordination in networked cyber-physical systems forces learning algorithms to simultaneously handle hybrid discrete-continuous actions, hard training-time safety constraints, and physics-governed dynamics. We show that these three features form a directed cycle of biases that defeats any naive composition of off-the-shelf modules, and formalize this as a three-way coupling lemma. We then introduce TRIDENT, the first MARL framework whose three components are co-designed to cancel each leak: a Richardson-Romberg gradient correction reducing Gumbel-Softmax bias from O(tau) to O(tau^2), a Lyapunov-constrained sequential trust-region update enforcing per-iterate feasibility, and a physics-informed residual critic that decomposes value rather than reward. We prove an O~(1/sqrt(K)) convergence rate to a constrained Nash equilibrium and an O(sqrt(K)) cumulative-violation bound. On multi-UAV mobile-edge computing, autonomous intersection management, and a hybrid SMAC variant, TRIDENT cuts training-time violations by 95.5% over MADDPG and 76.3% over MACPO, while improving reward by 13.5% over the strongest unconstrained baseline.

**arXiv ID:** 2606.18308
</details>

<details>
<summary><strong>Agentra: A Supervisable Multi-Agent Framework for Enterprise Intrusion Response</strong> - Raj Patel, Shaswata Mitra, Michele Guida, Stefano Iannucci, Sudip Mittal, Shahram Rahimi - [[pdf]](https://arxiv.org/pdf/2606.18325)</summary>

**Abstract:** Enterprise intrusion response still depends on static playbooks and analyst-driven triage, creating delay between alert generation and containment. We present Agentra, a supervisable multi-agent Intrusion Response System (IRS) framework that converts alerts from IDS, EDR, and XDR platforms into structured incident response plans grounded in MITRE ATT&CK, MITRE D3FEND, and NIST CSF 2.0. Agentra decomposes response reasoning across role-scoped agents, validates proposed plans through a bounded Planner--Validator review loop, screens retrieved threat intelligence through a Moderator security gateway, gates actions through an Action Catalog and risk score, and records decisions in an append-only audit log. We evaluate Agentra against a static OASIS CACAO v2.0 cyber-playbook baseline on a 120-event corpus drawn from ThreatHunter-Playbook, Splunk BOTSv3, and DARPA OpTC. The strongest configuration improves FP-aware IRS F1 from 0.61 to 0.84 and restores the projected harmful-action rate to the static baseline level of 0.0% after Planner-only configurations introduce unsafe overreaction. These results indicate that multi-agent response planning can improve ontology-grounded IRS coverage while preserving analyst approval and auditability.

**arXiv ID:** 2606.18325
</details>

<details>
<summary><strong>Skill-MAS: Evolving Meta-Skill for Automatic Multi-Agent Systems</strong> - Hehai Lin, Qi Yang, Chengwei Qin - [[pdf]](https://arxiv.org/pdf/2606.18837)</summary>

**Abstract:** Large Language Model (LLM)-based automatic Multi-Agent Systems (MAS) generation has become a crucial frontier for tackling complex tasks. However, existing methods face a dilemma between model capability and experience retention. Inference-time MAS leverages frozen frontier LLMs but repeats identical searches without learning from past experience. Conversely, Training-time MAS internalizes experience via gradient updates but is constrained by the low capability ceiling of smaller models, and is hard to scale to large frontier LLMs. To bridge this gap, we propose Skill-MAS, a novel third path that decouples experience retention from parametric updates by conceptualizing the high-level orchestration capability as an evolvable Meta-Skill. Skill-MAS refines this architectural knowledge through a closed optimization loop: (1) Multi-Trajectory Rollout samples a behavioral distribution for each task under the current Meta-Skill; and (2) Selective Reflection adaptively selects priority tasks and applies hierarchical contrastive analysis to distill systemic experience into generalizable, strategy-level principles. Extensive experiments across four complex benchmarks and four distinct LLMs demonstrate that Skill-MAS not only achieves remarkable performance gains but also maintains a favorable cost-performance trade-off. Further analysis reveals that the evolved Meta-Skills are highly robust and exhibit strong transferability across unseen tasks and different LLMs.

**arXiv ID:** 2606.18837
</details>

<details>
<summary><strong>CAPRA: Scaling Feedback on Software Architecture Deliverables with a Multi-Agent LLM System</strong> - Marco Becattini, Niccolò Caselli, Matteo Minin, Roberto Verdecchia, Enrico Vicario - [[pdf]](https://arxiv.org/pdf/2606.18976)</summary>

**Abstract:** Automated assessment in software engineering education has advanced significantly for code grading and essay scoring. However, reviewing software architecture deliverables, which requires analyzing structural completeness and requirements traceability, has not yet been fully automated. Applying Large Language Models (LLMs) to this task requires robust architectures to ensure technical feedback is accurate and reliable for students. This paper presents CAPRA (Configurable Architecture Proficiency Report Assessment), a multi-agent LLM system that analyzes software architecture deliverables to generate personalized, template-compliant LaTeX feedback. As a core design choice, CAPRA coordinates multiple specialized agents and employs a Python-based microservice for multi-modal document extraction, utilizing PyMuPDF and vision-enabled LLMs (specifically gpt-4o) to parse text and UML diagrams. To ensure educational reliability and mitigate hallucinations, CAPRA introduces a deterministic Evidence Anchoring step using fuzzy matching via normalized Levenshtein distance, along with a ConsistencyManager agent that cross-verifies, deduplicates, and merges findings. System performance is assessed using a structured eight-criterion binary evaluation taxonomy covering: (i) extraction completeness, (ii) feature validation, (iii) issue grounding and severity detection, (iv) recommendation specificity and traceability, and (v) template and tone compliance. A preliminary empirical evaluation on 10 student reports shows that CAPRA satisfied 88.8% of the evaluated criteria under a strict two-rater aggregation rule, achieved moderate inter-rater agreement with human evaluators (kappa = 0.582), and processed each report in slightly over 4 minutes. While these results support the viability of LLM-supported architectural feedback, human oversight remains essential for subjective assessment dimensions.

**arXiv ID:** 2606.18976
</details>

<details>
<summary><strong>Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams</strong> - Haewoon Kwak - [[pdf]](https://arxiv.org/pdf/2606.19111)</summary>

**Abstract:** Team science holds that leadership is contingent: it helps only under specific conditions, and capable, autonomous teams may need none at all. We ask the analogous question for multi-agent LLM teams: under what measurable conditions does process-level coordination control add value, and do those conditions match what team science predicts? We use behavioral signatures (majority lock-in, exploration, recovery from an incorrect round-0 consensus) and per-action ablations, clean because each controller is an explicit action set, not a monolithic prompt. We operationalize three classical leadership styles (transactional, transformational, situational) as controllers over a shared action vocabulary (explore, revise, accept, synthesize). A matched controller with the same actions but an arbitrary rule recovers no better than majority voting, so the theory-derived rule, not the vocabulary, does the work. Across four task regimes and three open-weight model families, no controller dominates by accuracy, as the contingency view predicts: transactional control matches a shared round-0 vote on all 12 (model, regime) combinations to within 1.3pp, and gains appear only on the one combination where the round-0 majority is unreliable (llama-4-scout social; situational +8pp over flat). A recovery-advantage account, tested with four boundary probes, says a controller beats plain interaction only where the round-0 majority is unreliable, the task is recoverable, and undirected interaction does not already repair it. These regions map onto contingency theory (leadership substitutes, path-goal redundancy, the situational readiness gap), so a largely null accuracy result is what the theory predicts, not a failure of the controllers. We read process-level coordination control as a contingency to be measured and theory-mapped, not a leaderboard to be topped.

**arXiv ID:** 2606.19111
</details>

<details>
<summary><strong>A Technical Taxonomy of LLM Agent Communication Protocols</strong> - Linus Sander, Habtom Kahsay Gidey, Alexander Lenz, Alois Knoll - [[pdf]](https://arxiv.org/pdf/2606.19135)</summary>

**Abstract:** As large language models (LLMs) advance and multi-agent systems aim to overcome the limits of standalone agents, robust communication protocols are becoming essential infrastructure for distributed agent networks. Nonetheless, the fragmented protocol landscape presents a significant interoperability challenge. This study develops a technical taxonomy to classify and analyze LLM agent communication protocols. Following an established iterative method, we defined the taxonomy's purpose, meta-characteristic, and ending conditions, then performed five iterations, three empirical-to-conceptual and two conceptual-to-empirical, on nine actively maintained open-source protocols with demonstrable adoption. The taxonomy comprises five dimensions: counterparty, payload, interaction state, discovery mechanism, and schema flexibility. Classification reveals recurring architectural patterns: all sampled agent-to-agent protocols combine hybrid payloads with session-state persistence; most protocols support multiple predefined schemas, and two negotiate schemas at runtime, indicating a trend toward schema flexibility; decentralized discovery remains rare. Analysis suggests short-term convergence pressure toward protocols unifying agent-to-agent and agent-to-context (tool and data) communication. Long-term, however, no single protocol is likely to maximize versatility, efficiency, and portability simultaneously. The field will more likely evolve toward a federated, layered protocol stack. The framework guides protocol selection and highlights open research gaps such as privacy and policy enforcement.}

**arXiv ID:** 2606.19135
</details>

<details>
<summary><strong>AdsMind: A Physics-Grounded Multi-Agent System for Self-Correcting Discovery of Adsorption Configurations on Heterogeneous Catalyst Surfaces</strong> - Zongmin Zhang, Yuyang Lou, Bowen Zhang, Junwu Chen, Ryo Kuroki, Xuan Vu Nguyen, Edvin Fako, Lixue Cheng, Philippe Schwaller - [[pdf]](https://arxiv.org/pdf/2606.19152)</summary>

**Abstract:** Identifying the lowest-energy surface-adsorbate configuration is critical for modeling heterogeneous catalysis, yet exhaustive exploration with ab initio calculations is computationally prohibitive. Machine-learning force fields (MLFFs) accelerate structural relaxation but leave the search over the vast configurational space a major bottleneck, and open-loop large language model (LLM) agents lack a physics-grounded feedback mechanism to correct erroneous initial guesses. We propose AdsMind (Adsorption configuration discovery with Machine intelligence and relaxation feedback), a closed-loop multi-agent framework that enables autonomous error correction through MLFF relaxation feedback. Across four LLM backends, AdsMind achieves consistently high search reliability, with success rates of 100% and 98.8% on the benchmarks AA20 and OCD-GMAE62. Relative to its single-pass (1-Shot) ablation it reduces cross-backend energy dispersion, and it uses only 4.11 and 4.67 MLFF relaxations per case, respectively -- an approximately 14-fold reduction over heuristic enumeration baselines. Density functional theory (DFT) validation using VASP/PBE on six representative AA20 systems shows that the reported open-loop Adsorb-Agent outputs exhibit qualitative adsorption-energy sign errors for molecular adsorbates, whereas AdsMind preserves the correct sign in all tested cases with closer quantitative agreement. AdsMind thus delivers reliability, self-reflection, and interpretability simultaneously, supporting more DFT-informed autonomous chemistry workflows.

**arXiv ID:** 2606.19152
</details>

<details>
<summary><strong>PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation</strong> - Jiho Choi, Seojeong Park, Seongjong Song, Hyunjung Shim - [[pdf]](https://arxiv.org/pdf/2508.21720)</summary>

**Abstract:** Automating scientific poster generation requires hierarchical document understanding and coherent content-layout planning. Existing methods often rely on flat summarization or optimize content and layout separately. As a result, they often suffer from information loss, weak logical flow, and poor visual balance. We present PosterForest, a training-free framework for scientific poster generation. Our method introduces the Poster Tree, a structured intermediate representation that captures document hierarchy and visual-textual semantics across multiple levels. Building on this representation, content and layout agents perform hierarchical reasoning and recursive refinement, progressively optimizing the poster from global organization to local composition. This joint optimization improves semantic coherence, logical flow, and visual harmony. Experiments show that PosterForest outperforms prior methods in both automatic and human evaluations, without additional training or domain-specific supervision.

**arXiv ID:** 2508.21720
</details>

<details>
<summary><strong>A Distributionally Robust Reinforcement Learning Framework for Constrained Urban EV Dispatch</strong> - An Nguyen, Hoang Nguyen, Phuong Le, Hung Pham, Cuong Do, Laurent El Ghaoui - [[pdf]](https://arxiv.org/pdf/2604.25848)</summary>

**Abstract:** We study city-scale control of electric-vehicle (EV) ride-hailing fleets where dispatch, repositioning, and charging decisions must respect charger and feeder limits under uncertain, spatially correlated demand and travel times. We formulate the problem as a hex-grid semi-Markov decision process (semi-MDP) with mixed actions -- discrete actions for serving, repositioning, and charging, together with continuous charging power -- and variable action durations. To guarantee physical feasibility during both training and deployment, the policy learns over high-level intentions produced by a masked, temperature-annealed actor. These intentions are projected at every decision step through a time-limited rolling mixed-integer linear program (MILP) that strictly enforces state-of-charge, port, and feeder constraints. To mitigate distributional shifts, we optimize a Soft Actor-Critic (SAC) agent against a Wasserstein-1 ambiguity set with a graph-aligned Mahalanobis ground metric that captures spatial correlations. The robust backup uses the Kantorovich-Rubinstein dual, a projected subgradient inner loop, and a primal-dual risk-budget update. Our architecture combines a two-layer Graph Convolutional Network (GCN) encoder, twin critics, and a value network that drives the adversary. Experiments on a large-scale EV fleet simulator built from NYC taxi data show that PD-RSAC achieves the highest net profit, reaching \$1.22M, compared with \$0.58M-\$0.70M for strong heuristic, single-agent RL, and multi-agent RL baselines, including Greedy, SAC, MAPPO, and MADDPG, while maintaining zero feeder-limit violations.

**arXiv ID:** 2604.25848
</details>

<details>
<summary><strong>Toward Vibe Medicine: A Self-Evolving Multi-Agent Framework for Clinical Decision Support</strong> - Qianxue Zhang, Yiming Ren, Shihuan Qin, Xiao Zhang, Liao Zhang, Jinyang Huang, Zhengliang Liu, Chenbin Liu, Hongying Feng, Jingyuan Chen, Yuzhen Ding, Weihang You, Hanqi Jiang, Yi Pan, Yifan Zhou, Junhao Chen, Lifeng Chen, Wei Liu, Tianming Liu, Zengren Zhao, Lian Zhang - [[pdf]](https://arxiv.org/pdf/2606.15504)</summary>

**Abstract:** In recent years, the advances of large language models and autonomous agents have revolutionized the healthcare field, facilitating diagnosis and improving treatment results. However, most existing AI systems rely on pre-trained knowledge and predefined pipelines, which struggle to learn dynamically from the interactive chat session history that contains patient outcomes and past failures. To address this limitation, we propose VIBEMed, a multi-agent framework with a built-in self-evolution mechanism and architecture-level safety sandbox for robust clinical decision support. The system integrates three specialized agents, including a Clinical Diagnostic Agent (CDA) for hypothesis generation, a Therapeutic Execution Agent (TEA) for treatment planning, and a Clinical Evolution Manager Agent (CEMA) that distills longitudinal clinical feedback into reusable knowledge, transforming multimodal patient information into personalized medical decisions. Through self-evolution mechanism, the framework enables iterative updates across memory, model behavior, and decision strategies, allowing the system to improve over time. Experimental results show that VIBEMed demonstrates superior performance through its evolving mechanism in complex clinical cases, particularly in tasks that require integrated decision-making and longitudinal planning. The framework also supports reliable end-to-end decisions in challenging scenarios such as oncology treatment planning, highlighting its feasibility in real-world clinical contexts. Overall, VIBEMed provides a practical path beyond static AI systems toward adaptive, experience-driven clinical decision support, demonstrating the value of combining multi-agent collaboration with continuous evolution for advancing precision medicine.

**arXiv ID:** 2606.15504
</details>

<details>
<summary><strong>Self-Evolving Multi-Agent Systems via Textual Backpropagation</strong> - Xiaowen Ma, Yunpu Ma, Chenyang Lin, Sikuan Yan, Jinhe Bi, Zixuan Cao, Yijun Tian, Volker Tresp, Hinrich Schuetze - [[pdf]](https://arxiv.org/pdf/2506.09046)</summary>

**Abstract:** Leveraging multiple Large Language Models (LLMs) has proven effective for addressing complex, high-dimensional tasks, but current approaches often rely on static, manually engineered multi-agent configurations. To overcome these constraints, we present the Agentic Neural Network (ANN), a framework that conceptualizes multi-agent collaboration as a layered neural network architecture. In this design, each agent operates as a node, and each layer forms a cooperative team focused on a specific subtask. Our framework follows a two-phase optimization strategy: (1) Forward Phase - Drawing inspiration from neural network forward passes, tasks are dynamically decomposed into subtasks, and cooperative agent teams with suitable aggregation methods are constructed layer by layer. (2) Backward Phase - Mirroring backpropagation, we refine both global and local collaboration through iterative feedback, allowing agents to self-evolve their roles, prompts, and coordination. This neuro-symbolic approach enables our framework to create new or specialized agent teams post-training, delivering notable gains in accuracy and adaptability. Across seven benchmark datasets, our work surpasses leading multi-agent baselines under the same configurations, showing consistent performance improvements.

**arXiv ID:** 2506.09046
</details>

<details>
<summary><strong>R2BC: Multi-Agent Imitation Learning from Single-Agent Demonstrations</strong> - Connor Mattson, Varun Raveendra, Ellen Novoseller, Nicholas Waytowich, Vernon J. Lawhern, Daniel S. Brown - [[pdf]](https://arxiv.org/pdf/2510.18085)</summary>

**Abstract:** Imitation Learning (IL) is a natural way for humans to teach robots, particularly when high-quality demonstrations are easy to obtain. While IL has been widely applied to single-robot settings, relatively few studies have addressed the extension of these methods to multi-agent systems, especially in settings where a single human must provide demonstrations to a team of collaborating robots. In this paper, we introduce and study Round-Robin Behavior Cloning (R2BC), a method that enables a single human operator to effectively train multi-robot systems through sequential, single-agent demonstrations. Our approach allows the human to teleoperate one agent at a time and incrementally teach multi-agent behavior to the entire system, without requiring demonstrations in the joint multi-agent action space. We show that R2BC methods match, and in some cases surpass, the performance of an oracle behavior cloning approach trained on privileged synchronized demonstrations across four multi-agent simulated tasks. Finally, we deploy R2BC on two physical robot tasks trained using real human demonstrations.

**arXiv ID:** 2510.18085
</details>

<details>
<summary><strong>DeepInflation: an AI agent for research and model discovery of inflation</strong> - Ze-Yu Peng, Hao-Shi Yuan, Qi Lai, Jun-Qian Jiang, Gen Ye, Jun Zhang, Yun-Song Piao - [[pdf]](https://arxiv.org/pdf/2601.14288)</summary>

**Abstract:** We present DeepInflation, an AI agent designed for research and model discovery in inflationary cosmology. Built upon a multi-agent architecture, DeepInflation integrates Large Language Models (LLMs) with a symbolic regression (SR) engine and a retrieval-augmented generation (RAG) knowledge base. This framework enables the agent to automatically explore and verify the vast landscape of inflationary potentials while grounding its outputs in established theoretical literature. We demonstrate that DeepInflation can successfully discover simple and viable single-field slow-roll inflationary potentials consistent with the latest observations (with the ACT DR6 results taken as an example) or any given $n_s$ and $r$, and provide accurate theoretical context for obscure inflationary scenarios. DeepInflation serves as a prototype for a new generation of autonomous scientific discovery engines in cosmology, which enables researchers and non-experts alike to explore the inflationary landscape using natural language. This agent is available at this https URL.

**arXiv ID:** 2601.14288
</details>

<details>
<summary><strong>PersonalPlan: Planning Multi-Agent Systems for Personalized Programming Learning</strong> - Zhiyuan Wen, Jiannong Cao, Peng Gao, Haochen Shi, Wengpan Kuan, Bo Yuan, Xiuxiu Qi - [[pdf]](https://arxiv.org/pdf/2606.18633)</summary>

**Abstract:** Effective programming education requires personalized instruction adapted to diverse learner backgrounds. However, while LLM-based multi-agent systems (MAS) excel at complex planning, existing planners often lack profile-grounding and pedagogical scaffolding, thereby undermining personalized programming learning. To fill in the gap, we first introduce \textbf{MAP-PPL} (\textbf{M}ulti-\textbf{A}gent \textbf{P}lans for \textbf{P}ersonalized \textbf{P}rogramming \textbf{L}earning), a profile-conditioned multi-agent planning dataset with 3{,}043 query--profile--plan instances from 1{,}730 Stack Overflow question groups and 2{,}738 learner profiles. Each plan specifies agents, subtasks, executable steps, and prerequisite dependencies. Then, we propose \textbf{PersonalPlan}, a two-stage MAS planner that first performs hierarchical SFT with separate LoRA adapters for profile-aware task decomposition and step dependency planning, then applies a Reward-Adaptive GRPO to encourage the model to generate executable, personalized, and pedagogically scaffolded plans. Extensive experiments on MAP-PPL comparing PersonalPlan against frontier LLMs, generic MAS frameworks, and agentic planners demonstrate its superiority. With only 8B and 32B variants, PersonalPlan achieves state-of-the-art plan executability, personalization, and pedagogical quality, effectively orchestrating MAS for agent-student interactions.

**arXiv ID:** 2606.18633
</details>

<details>
<summary><strong>EARS: Explanatory Abstention for Reliable Sub-Agent Modeling in Large-scale Multi-Agent Systems</strong> - Shuang Xie, Yunan Lu, Han Li, Lingyun Wang - [[pdf]](https://arxiv.org/pdf/2606.18668)</summary>

**Abstract:** In large-scale enterprise settings, centralized multi-agent systems (MAS) are increasingly adopted, in which a coordinator delegates user requests to lightweight, domain-specialized sub-agents. While this architecture improves modularity, scalability, and cost efficiency, its reliability depends not only on accurate routing but also on sub-agents' ability to calibrate their responses to capability constraints. In particular, sub-agents built on smaller fine-tuned models often struggle with such calibration, leading them to over-answer ambiguous, underspecified, misrouted, or unsupported requests and produce hallucinated outputs instead of actionable feedback. To address this challenge, we present EARS (Explanatory Abstention for Reliable Sub-Agent Modeling), a production-oriented framework that reframes sub-agent abstention as an inter-agent communication protocol: a sub-agent does not merely abstain, but exposes an actionable failure state to the coordinator. EARS curates human-agent interaction data using an ensemble of calibrated LLM-as-a-Judge models, producing structured abstention labels and rationales under a taxonomy of sub-agent failure modes. These data are used to fine-tune sub-agents to detect failure conditions and return rationales for coordinator-level clarification, rerouting, or fallback. We evaluate EARS in a large-scale production e-commerce assistant supporting enterprise business intelligence workflows. EARS improves the overall response pass rate from 68.5% to 78.9%, demonstrating that sub-agent-side explanatory abstention improves MAS reliability.

**arXiv ID:** 2606.18668
</details>

<details>
<summary><strong>Enhancing Decision-Making with Large Language Models through Multi-Agent Fictitious Play</strong> - Leyang Shen, Yang Zhang, Xiaoyan Zhao, Chun Kai Ling, Tat-Seng Chua - [[pdf]](https://arxiv.org/pdf/2606.19308)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems (MAS) have demonstrated great potential in solving tasks with execution complexity, by distributing subtasks across cooperative agents. However, this divide-and-conquer paradigm falls short on decision-making tasks that are also prevalent in the real world. These tasks require simultaneous reasoning from the stances of all involved stakeholders whose decisions are mutually dependent and thus cannot be solved in isolation. We characterize this challenge as stance entanglement, a form of decision complexity distinct from execution complexity. To address it, we propose Multi-Agent Fictitious Play (MAFP), a novel MAS paradigm that represents stakeholder stances as agents and formulates decision-making as an equilibrium-seeking process. Built on the game-theoretic principle of fictitious play, MAFP iteratively updates each agent's decision by best responding to the empirical mixture of other agents' past decisions. This enables agents to expose and address one another's weaknesses, progressively improving decision quality and robustness. We evaluate MAFP on challenging decision-making tasks that test the capability of deciding strategies for competitive scenarios prior to acting. MAFP outperforms both single-round and multi-round baselines on two complementary metrics, tournament strength and robustness, demonstrating its effectiveness in addressing stance entanglement.

**arXiv ID:** 2606.19308
</details>

<details>
<summary><strong>Epistemic Gain, Aleatoric Cost: Uncertainty Decomposition in Multi-Agent Debate for Math Reasoning</strong> - Dan Qiao, Binbin Chen, Fengyu Cai, Jianlong Chen, Wenhao Li, Fuxin Jiang, Zuzhi Chen, Hongyuan Zha, Tieying Zhang, Baoxiang Wang - [[pdf]](https://arxiv.org/pdf/2603.01221)</summary>

**Abstract:** Multi-Agent Debate (MAD) has shown promise in improving reasoning and reducing hallucinations, yet it remains unclear how information exchange shapes individual reasoning behavior. Empirically, MAD exhibits paradoxical phenomena, including rising accuracy with increasing token entropy and marked differences between homogeneous and heterogeneous agent combinations. In this paper, we introduce a Bayesian uncertainty analysis framework for MAD, which decomposes answer-level predictive uncertainty into epistemic uncertainty and aleatoric uncertainty, corresponding to the potential gain and cost of debate. Across multiple agent configurations, we find that effective debate depends on achieving high epistemic gain under controlled aleatoric cost. Building on this insight, we design an uncertainty-guided multi-agent reinforcement learning algorithm that encourages lower aleatoric cost and more effective epistemic information utilization. Experiments show that our approach simultaneously enhances each agent's accuracy and promotes a more productive debate process, providing an operational Bayesian perspective for understanding and improving MAD.

**arXiv ID:** 2603.01221
</details>

<details>
<summary><strong>Multi-Agent Systems are Mixtures of Experts: Who Becomes an Influencer?</strong> - Franka Bause, Jonas Niederle, Martin Pawelczyk, Rebekka Burkholz - [[pdf]](https://arxiv.org/pdf/2605.25929)</summary>

**Abstract:** The effectiveness of multi-agent LLM deliberation depends not only on the agents' individual predictions, but also on how they communicate and collaborate. We study this mechanism through the lens of Friedkin-Johnsen (FJ) opinion dynamics, a tractable model for analyzing stubbornness, influence, and opinion change in multi-agent systems that captures empirically observed deliberation patterns. We show that the FJ parameters are input-dependent, turning multi-agent deliberation into a mixture of experts. This perspective implies that multi-agent systems can outperform single agents and static ensembles when routing reflects agent competence. Since competence is latent in practice, we analyze how influence is established through observable proxies: agents' self-assessed confidence, their perceived confidence, and initial alignment with other agents' views.

**arXiv ID:** 2605.25929
</details>

<details>
<summary><strong>From Privacy to Workflow Integrity: Communication-Graph Metadata in Autonomous Agent Interoperability</strong> - Bijaya Dangol - [[pdf]](https://arxiv.org/pdf/2606.07150)</summary>

**Abstract:** Agent-interoperability protocols such as A2A and MCP standardize what agents say to one another but assume address-based transport. Whether over HTTP(S) or a content-protecting binding such as MLS-based SLIM, these transports protect message content yet leave the communication graph exposed: which agent contacts which, when, and how often. In agent systems this graph is more consequential than a privacy framing suggests. Endpoints are capability-labeled, workflows are structured and chained, and interactions are coupled to actions, so an observer recovers more than past relationships: it can recognize a recurring pending workflow from its opening and, at machine speed, act on it before it completes. The threat is one of workflow integrity, not privacy alone. We give a threat model for the communication graph and locate what makes its metadata distinctively consequential: not stronger fingerprinting but exposure across independent trust domains, coupled to autonomous action. We define transport- and bootstrap-layer privacy properties, give them an indistinguishability-game semantics, evaluate transports, and give an A2A case study where a metadata-protecting binding surfaces its implicit identity assumptions. On a corpus of real multi-agent A2A traffic from the official reference agents, on a live A2A binding, and with a generative model as a controlled instrument, a label-blind classifier recovers a task's class from passive metadata at 6x chance, and from only its opening; a defense-aware adversary does not overturn this, and only the full set of properties drives recovery toward chance. Acting on the leak is distinct from recoverability: under a fixed budget an adversary captures 0.63 of a clairvoyant attacker's advantage on the corpus (0.41 from a workflow's opening), governed by top-ranked precision rather than overall accuracy, so integrity and privacy come apart under defense.

**arXiv ID:** 2606.07150
</details>

<details>
<summary><strong>Towards Scalable Customization and Deployment of Multi-Agent Systems for Enterprise Applications</strong> - Paresh Dashore, Shreyas Kulkarni, Uttam Gurram, Nadia Bathaee, Kartik Balasubramaniam, Genta Indra Winata, Sambit Sahu, Shi-Xiong Zhang - [[pdf]](https://arxiv.org/pdf/2606.18502)</summary>

**Abstract:** Large language model (LLM)-based multi-agent systems demonstrate strong performance on complex reasoning and task execution, enabling broad enterprise applications. However, production deployment remains challenging due to domain-specific customization requirements and high latency and inference costs in agentic workflows. We propose a unified framework for customization and efficient deployment of multi-agent systems in real-world settings. The first stage, Agentic Model Customization, combines continual pretraining, supervised fine-tuning, and preference optimization to adapt a compact model to specialized domains while retaining strong agentic capabilities. The second stage, Inference Optimization, integrates speculative decoding and FP8 quantization with targeted calibration to enable cost-efficient serving with minimal quality loss. Across enterprise workloads, our framework enables rapid domain adaptation and achieves a 4.48x speedup in throughput while maintaining performance and improving robustness on long-tail scenarios.

**arXiv ID:** 2606.18502
</details>

<details>
<summary><strong>SAGE: Stochastic Prompt Optimization via Agent-Guided Exploration</strong> - Ziyi Zhu, Luka Smyth, Saki Shinoda, Jinghong Chen - [[pdf]](https://arxiv.org/pdf/2606.18902)</summary>

**Abstract:** Context engineering has emerged as a primary lever for improving AI systems without parameter updates. Recent work showing that textual gradients do not function as real gradients motivates treating automatic prompt optimization (APO) as black-box search. We introduce SPO (Stochastic Prompt Optimization), a framework for stochastic search over prompt space, and compare three strategies of increasing sophistication: error-informed random search, a genetic algorithm with evolutionary operators, and SAGE (SPO via Agent-Guided Exploration), a multi-agent pipeline with diagnostic code execution. Across three benchmarks, no single strategy dominates; effectiveness depends on the interaction of landscape structure with error type. We further deploy SAGE on a mental-health chatbot under a continuous optimization paradigm, where it compounds eight cycles of individually-noisy A/B tests into a statistically robust gain in next-day retention. We argue that coupling qualitative diagnosis with quantitative validation is what makes agentic optimization effective for open-ended task-oriented dialogue.

**arXiv ID:** 2606.18902
</details>

<details>
<summary><strong>Do as the Romans Do: Learning Universal Behaviors from Heterogeneous Agents</strong> - Caleb Chang, Davin Win Kyi, Natasha Jaques, Karen Leung - [[pdf]](https://arxiv.org/pdf/2606.18537)</summary>

**Abstract:** Humans often acquire new skills by observing others, since observed behaviors implicitly reveal how to act in an environment. However, observations drawn from a heterogeneous population introduce conflicting behavioral signals, making it difficult to determine which behaviors are worth imitating. We address this challenge with General Reward Inference and Disentanglement (GRID), a social learning method that extracts universally useful behaviors from a heterogeneous population of demonstrators pursuing different goals. GRID decomposes per-agent reward functions into a general reward, capturing behaviors shared across all agents, and specific rewards, capturing individual preferences and objectives. Training exclusively on the general reward provides a new paradigm of generalist pretraining. It yields a generalist agent that internalizes universal environmental competencies, such as safety and basic task proficiency, without the mode-averaging bias that afflicts standard learning from demonstration techniques. This generalist serves as a superior prior for fine-tuning to downstream tasks, including preferences unseen during training. Experiments across a synthetic basis function decomposition, multi-agent Craftax, and a continuous autonomous driving simulator (Highway-Env) confirm that GRID successfully disentangles reward structure in a semantically meaningful way, outperforms standard learning from demonstration baselines, and enables more efficient and stable specialization.

**arXiv ID:** 2606.18537
</details>

<details>
<summary><strong>A Mixed-Reality Testbed for Autonomous Vehicles</strong> - H. M. Sabbir Ahmad, Ehsan Sabouni, Emrullah Celik, Zean Wan, Damola Ajeyemi, Christos G. Cassandras, Wenchao Li - [[pdf]](https://arxiv.org/pdf/2606.19267)</summary>

**Abstract:** We propose a mixed-reality, hardware-in-the-loop (HIL) testbed for autonomous vehicles that seamlessly integrates a physical testbed of mobile robots with a high-fidelity simulation environment. The virtual simulation enables the creation of diverse, safety-critical driving scenarios to validate state-of-the-art perception, planning, and control algorithms, while augmenting simulations with physical robots equipped with multimodal sensors in photorealistic virtual environments further facilitating rigorous validation. Our testbed also features vehicular connectivity using wireless communication and can accommodate a large number of agents through the combination of physical robots and virtual simulated agents, supporting research on multi-agent systems including Connected and Autonomous Vehicles (CAVs). Finally, we present a safety-guaranteed framework combining perception, planning and a novel online learning-based controller using Control Barrier Functions (CBFs) for CAVs. Experiments using the proposed framework are used to validate and demonstrate the key functionalities and the overall utility of the testbed to bridge the gap between simulation and real-world hardware deployment.

**arXiv ID:** 2606.19267
</details>

</details>

<details open>
<summary><h2>Other Agent Research (6 papers)</h2></summary>

<details>
<summary><strong>Skill-Guided Continuation Distillation for GUI Agents</strong> - Zhimin Fan, Hongwei Yu, Yeqing Shen, Haolong Yan, Guozhen Peng, Tianhao Peng, Yudong Zhang, Xiaowen Zhang, Kaijun Tan, Zheng Ge, Xiangyu Zhang, Daxin Jiang - [[pdf]](https://arxiv.org/pdf/2606.18890)</summary>

**Abstract:** Improving GUI agents typically relies on behavior cloning on expert trajectories. However, as the current policy deviates from the expert policy, it inevitably encounters policy-induced off-trajectory states during closed-loop execution, i.e., states that fall outside the expert trajectories. Since expert trajectories provide no demonstrations for these unseen states, such states receive no effective supervision, leaving the policy unable to select the correct action. To close this supervision gap, we propose Skill-Guided Continuation Distillation (SGCD), an iterative self-improvement framework. SGCD first runs the plain policy without skill guidance for a few steps to reach realistic off-trajectory states. From these states, a skill-guided policy then completes the task and produces successful continuations, which are mixed with expert trajectories to supply supervision over policy-induced off-trajectory states. The skills are extracted from both successful and failed rollouts, consisting of Continuation Plans, Critical Targets, Failure Traps, and Success Criteria. On OSWorld-Verified, SGCD improves the success rate of three base models from the low-30\% range to over 50\%, demonstrating its effectiveness and generality.

**arXiv ID:** 2606.18890
</details>

<details>
<summary><strong>Towards an Agent-First Web: Redesigning the Web for AI Agents</strong> - Eranga Bandara, Ross Gore, Ravi Mukkamala, Asanga Gunaratna, Safdar H. Bouk, Xueping Liang, Peter Foytik, Abdul Rahman, Sachini Rajapakse, Isurunima Kularathna, Pramoda Karunarathna, Chalani Rajapakse, Ng Wee Keong, Kasun De Zoysa, Tharaka Hewa, Amin Hass, Wathsala Herath, Aruna Withanage, Nilaan Loganathan, Atmaram Yarlagadda, Sachin Shetty - [[pdf]](https://arxiv.org/pdf/2606.19116)</summary>

**Abstract:** The World Wide Web was built on an assumption held for three decades: the primary consumer of web content is a human being. This permeates every layer; its access model presumes human visitors, its economics rest on human attention, and its content targets human perception. The rapid emergence of AI agents as intermediaries between humans and web content invalidates this assumption. Yet the web resists agents through blanket blocking, CAPTCHA-based exclusion, and economic models that treat agent access as extraction rather than legitimate interaction.
This paper proposes a principled redesign across three layers. At the access layer, agents acting for humans should inherit equivalent access rights, governed by rate limiting and agent identification metadata in HTTP requests, analogous to browser headers, alongside a dual-layer architecture serving human-readable and agent-optimized content from the same domain. At the economic layer, we propose an intent-based tier framework grounded in the agent-as-human-proxy principle: an agent's economic obligation mirrors that of the human it represents. A token-based subscription model meters content in tokens rather than pageviews, alongside a commissioned content economy anchoring AI content production in human intentionality. At the content layer, we identify epistemic recursion, the self-referential loop in which AI-generated content is consumed by agents to produce further content, progressively detaching web knowledge from human ground truth. We propose the Agent Text Markup Language (ATML), a four-level human supervision tier model, and a cryptographic provenance chain to counter this threat.
Together these constitute ten design principles for an agent-first internet, one in which agents are first-class citizens whose integration requires renegotiating the web's foundational social contract across access, economics, and content.

**arXiv ID:** 2606.19116
</details>

<details>
<summary><strong>Caring Without Feeling: Affective Dynamics as the Control Layer of Human-AI Agent Collaboration</strong> - Junjie Xu, Xingjiao Wu, Zihao Zhang, Yujia Xu, Yuzhe Yang, Jin Zhu, Luwei Xiao, Wen Wu, Liang He - [[pdf]](https://arxiv.org/pdf/2606.18259)</summary>

**Abstract:** AI agents that plan, retain memory across sessions, invoke external tools and act with partial autonomy are transforming human--AI collaboration. Research on affective computing, simulated empathy in large language models, trust in automation and AI safety has illuminated important design principles, yet these literatures remain fragmented. No integrated account explains how affective cues operate within agentic collaboration -- settings in which humans delegate, monitor and correct consequential tasks. This Review synthesises computational and interactional mechanisms of affective dynamics: the processes through which affective cues, emotion-like behaviour and perceived agent affect shape trust calibration, delegation decisions, error correction, dependence and governance. We trace how model-generated affective signals enter interaction loops that govern reliance, repair and oversight, and propose a framework that treats affect not as an internal property of AI but as a coordination layer through which humans and agents negotiate capability, uncertainty and responsibility. The framework provides a foundation for calibrated measurement, purposeful design and informed governance.

**arXiv ID:** 2606.18259
</details>

<details>
<summary><strong>Guava: An Effective and Universal Harness for Embodied Manipulation</strong> - Haowen Liu, Xirui Li, Shaoxiong Yao, Peng Shi, Tianyi Zhou, Jia-Bin Huang, Furong Huang, Jiayuan Mao - [[pdf]](https://arxiv.org/pdf/2606.18363)</summary>

**Abstract:** Language models trained on large-scale vision-language data have demonstrated strong potential for embodied agents. Harnessing models through embodied tools use offers a promising alternative to end-to-end vision-language-action systems by combining high-level reasoning with external modules for perception, planning, and control. However, it remains unclear what makes an effective harness for embodied manipulation, and to what extent such a harness can unlock embodied capabilities in a wide range of reasoning models. In this work, we present Guava, a harness framework for embodied tool use developed through systematic exploration of the design space of agent workflows, action spaces, and observation spaces. Our study identifies three key ingredients for effective embodied agents: iterative perception-reasoning-action loops, semantic action abstractions, and multimodal observations. To understand whether these design principles are universal even to small models, we develop an end-to-end training pipeline that distills embodied manipulation capabilities into a 4B open-source model using fewer than 2K trajectories collected entirely in simulation. Experimental results in both simulation and real-world environments show performance comparable to frontier proprietary models while exhibiting strong generalization to unseen objects, novel instructions, and long-horizon tasks. Results suggest that a well-designed harness can serve as a scalable, model-agnostic interface for embodied manipulation, enabling strong emergent embodied capabilities in compact open-source models with minimal training data.

**arXiv ID:** 2606.18363
</details>

<details>
<summary><strong>Hardware- and Vision-in-the-Loop Validation of Deep Monocular Pose Estimation for Autonomous Maritime UAV Flight</strong> - Maneesha Wickramasuriya, Beomyeol Yu, Jaden Shin, Mason Huslig, Taeyoung Lee, Murray Snyder - [[pdf]](https://arxiv.org/pdf/2606.19176)</summary>

**Abstract:** Autonomous UAV operations on ships require reliable vision-based relative pose estimation, yet at-sea validation is costly, weather-dependent, and risky. This paper presents a hardware-validated vision-in-the-loop framework that enables fully autonomous indoor flight while emulating photorealistic maritime environments. Rendered maritime views are processed onboard by a deep transformer-based monocular pose estimator. Delayed vision measurements are fused with high-rate IMU data using a delayed Kalman filter to provide consistent state estimates for geometric control. The system captures critical embedded effects, including perception latency, asynchronous updates, and computational constraints, that are absent in pure simulation. Autonomous takeoff, trajectory tracking, and landing experiments demonstrate stable closed-loop flight. The results establish a safe and hardware-realistic intermediate stage for developing maritime UAV autonomy prior to shipboard deployment.

**arXiv ID:** 2606.19176
</details>

<details>
<summary><strong>MORTAR: Multi-turn Metamorphic Testing for LLM-based Dialogue Systems</strong> - Aaron Guoxiang Guo, Aldeida Aleti, Neelofar Neelofar, Chakkrit Tantithamthavorn, Yuanyuan Qi, Tsong Yueh Chen - [[pdf]](https://arxiv.org/pdf/2412.15557)</summary>

**Abstract:** With the widespread application of LLM-based dialogue systems in daily life, quality assurance has become more important than ever. Recent research has successfully introduced methods to identify unexpected behaviour in single-turn testing scenarios. However, multi-turn interaction is the common real-world usage of dialogue systems, yet testing methods for such interactions remain underexplored. This is largely due to the oracle problem in multi-turn testing, which continues to pose a significant challenge for dialogue system developers and researchers. In this paper, we propose MORTAR, a metamorphic multi-turn dialogue testing approach, which mitigates the test oracle problem in testing LLM-based dialogue systems. MORTAR formalises the multi-turn testing for dialogue systems, and automates the generation of question-answer dialogue test cases with multiple dialogue-level perturbations and metamorphic relations (MRs). The automated MR matching mechanism allows MORTAR more flexibility and efficiency in metamorphic testing. The proposed approach is fully automated without reliance on LLM judges. In testing six popular LLM-based dialogue systems, MORTAR reaches significantly better effectiveness with over 150\% more bugs revealed per test case when compared to the single-turn metamorphic testing baseline. Regarding the quality of bugs, MORTAR reveals higher-quality bugs in terms of diversity, precision and uniqueness. MORTAR is expected to inspire more multi-turn testing approaches, and assist developers in evaluating the dialogue system performance more comprehensively with constrained test resources and budget.

**arXiv ID:** 2412.15557
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (21 papers)</h2></summary>

<details>
<summary><strong>What Must Generalist Agents Remember?</strong> - Khurram Yamin, Namrata Deka, Maitreyi Swaroop, Albert Ting, Jeff Schneider, Bryan Wilder - [[pdf]](https://arxiv.org/pdf/2606.18746)</summary>

**Abstract:** This paper develops a formal account of what generalist agents must store in memory in order to act near-optimally across multiple environments and goals. It shows that when two domains share an observational bottleneck but require incompatible optimal actions, any uniformly near-optimal policy must induce distinct memory distributions at that bottleneck. The result yields a separation theorem: sufficiently successful agents cannot rely only on current state observations, but must preserve domain-relevant information in memory. The paper further shows that if an agent's memory contains enough information to estimate values for related goals, then that memory can be used to approximately reconstruct the agent's local transition dynamics. Together, these results characterize memory as the substrate that supports domain disambiguation, transition-model reconstruction, and planning for generalist agents.

**arXiv ID:** 2606.18746
</details>

<details>
<summary><strong>ThinkDeception: A Progressive Reinforcement Learning Framework for Interpretable Multimodal Deception Detection</strong> - Jinhao Song, Shan Liang, Yiqun Yue, Zhuhuayang Zhang, Tianqi Gao - [[pdf]](https://arxiv.org/pdf/2606.18988)</summary>

**Abstract:** Multimodal deception detection is critical for identifying fraudulent intentions, yet existing approaches predominantly rely on end to end black--box paradigms. These methods suffer from a severe lack of interpretability failing to provide transparent reasoning trajectories and struggling to explicitly capture the subtle, cross modal inconsistencies inherent in deceptive behaviors. To transcend these limitations, we propose ThinkDeception, a novel and interpretable multimodal deception detection framework. As a pioneering effort, it introduces Multimodal Large Language Models (MLLMs) into this domain, transforming deception detection from a traditional binary classification task into an explicit cognitive reasoning process. Facilitated by the first meticulously annotated step--by--step multimodal Chain of Thought (CoT) dataset, we develop a foundational model, ThinkDeception Base, empirically validating the critical role of modal inconsistency in decoding deception. Building upon this foundation, our core innovation lies in proposing Visual-Audio Consistency Group Relative Policy Optimization(VAC--GRPO) equipped with a progressive training strategy. Distinct from standard GRPO, we stratify the training data into four progressive difficulty tiers, guiding the model through a psychologically grounded easy--to--hard cognitive transition. By innovatively coupling this dynamic curriculum scheduler with a multi dimensional, process aware reward mechanism and a reflective learning paradigm, we significantly elevate the model's overall reasoning quality. Extensive experiments on mainstream benchmarks demonstrate that ThinkDeception establishes a new SOTA, significantly outperforming existing methods in both detection accuracy and rationale quality. Ultimately, this work successfully drives the field of deception detection toward interpretable, multimodal cognitive reasoning.

**arXiv ID:** 2606.18988
</details>

<details>
<summary><strong>RODS: Reward-Driven Online Data Synthesis for Multi-Turn Tool-Use Agents</strong> - Ruishan Fang, Siyuan Lu, Chenyi Zhuang, Tao Lin - [[pdf]](https://arxiv.org/pdf/2606.19047)</summary>

**Abstract:** Multi-turn tool-use RL is bottlenecked by the rapid depletion of informative samples in static datasets. We observe that the gradient signal in GRPO concentrates on tasks with the highest rollout reward variance, a consequence of the Popoviciu upper bound. Consequently, samples near the agent's capability boundary -- where successes and failures are roughly balanced -- contribute disproportionately large policy gradients. As training progresses, this boundary continuously shifts, which gradually depletes the pool of informative samples in a static dataset. We propose RODS (Reward-driven Online Data Synthesis) to resolve this depletion. RODS closes the loop between RL training and data generation by repurposing the progress reward variance as a practical, zero-cost boundary detector that requires no extra inference beyond the rollouts already computed for training. It continuously identifies such boundary samples, synthesizes new multi-turn variants matching their structural complexity (e.g., API topology and dependency depth) via a skill-aligned resampling pipeline, and manages a dynamic replay buffer that co-evolves with the policy. Starting from 400 human seeds and maintaining an active training pool of ~800 samples, RODS achieves comparable performance to a 17K-sample offline pipeline while requiring roughly 20x fewer trajectories, and improves over fixed-data RL and environment augmentation in our controlled setting.

**arXiv ID:** 2606.19047
</details>

<details>
<summary><strong>Self-CTRL: Self-Consistency Training with Reinforcement Learning</strong> - Itamar Pres, Laura Ruis, Melat Ghebreselassie, Belinda Z. Li, Jacob Andreas - [[pdf]](https://arxiv.org/pdf/2606.18327)</summary>

**Abstract:** Language models (LMs) that faithfully describe their own behavior can more easily be audited, understood, and trusted by users. This paper describes Self-Consistency Training with Reinforcement Learning (Self-CTRL), a method that optimizes for consistency between a LM's self-explanations and behavior on related inputs by updating explanations to better predict behavior or updating behavior to better match explanations. We apply our method in two domains. First, we study a formal probabilistic reasoning task in which LMs must learn to imitate a family of biased samplers and evaluated on their ability to report the associated biases. We find that consistency training improves the correlation between self-reported and behaviorally-measured latent biases from $R^2=0.24$ to $R^2=0.64$ on a set of held-out distributions, matching the generalization of direct ground-truth supervision. Second, we study a constitutional AI domain in which LMs must describe when they will refuse or comply with user requests. Here, Self-CTRL produces rules that faithfully describe the model's behavior on held-out requests, improving the refusal predictions of a third-party auditor model from $36\%$ to $92\%$. In the other direction, behavior updates improve alignment, reducing HarmBench failure rate from $15.0\%$ to $0.5\%$ without substantially increasing refusal on harmless prompts. By aligning explanations and behavior, our work provides a general recipe for training AI models to be safer, more transparent, and more controllable.

**arXiv ID:** 2606.18327
</details>

<details>
<summary><strong>Benchmarking Action Spaces in Reinforcement Learning for Vision-based Robotic Manipulation</strong> - Seyed Alireza Azimi, Homayoon Farrahi, Abhishek Naik, Colin Bellinger, A. Rupam Mahmood - [[pdf]](https://arxiv.org/pdf/2606.18594)</summary>

**Abstract:** In real-world reinforcement learning (RL), the choice of action space can play a key role in shaping motion smoothness, safety, and overall task performance. In this study, we evaluate pose increment, pose velocity, joint position increment, and joint velocity across two vision-based manipulation tasks: object picking and pushing. We train policies in simulation and deploy them to the real world using sim-to-real transfer. We find that action-space representation indeed significantly affects sim-to-real performance. In particular, we find that the joint velocity action space is best for the vision-based picking and pushing tasks in terms of smoothness and final task performance. We also provide practical guidance for RL practitioners in choosing action spaces for both simulation and real-world experiments.

**arXiv ID:** 2606.18594
</details>

<details>
<summary><strong>Generating Natural and Expressive Robot Gestures through Iterative Reinforcement Learning with Human Feedback using LLMs</strong> - Chris Lee, Flora Salim, Benjamin Tag, Francisco Cruz - [[pdf]](https://arxiv.org/pdf/2606.18747)</summary>

**Abstract:** Expressive gestures are essential for natural and effective communication, complementing speech when verbal cues alone are insufficient (e.g., pointing). For social robots such as the humanoid Pepper, producing natural and expressive movements is critical for improving human-robot interaction (HRI) and long-term acceptance. However, generating gestures remains challenging due to reliance on expert-authored animations, resulting in rigid behaviors that are impractical for dynamic and diverse environments. Alternatively, machine learning approaches often struggle to capture perceived naturalness, becoming increasingly challenging with more degrees of freedom. Consequently, producing expressive robot gestures requires a system that can adapt to the environment while adhering to social norms and physical constraints. Recent advances in large language models (LLMs) enable dynamic code generation, offering new opportunities for runtime gesture synthesis from natural language. In this paper, we integrate ChatGPT into the humanoid robot Pepper to generate co-speech gestures aligned with conversational output. While this baseline enables flexible gesture generation, the resulting motions are often perceived as stiff and unnatural. To address this limitation, we introduce an iterative reinforcement learning with human feedback (RLHF) system that finetunes gesture generation based on user evaluations, leveraging an iterative user study to compare Pepper's generated gestures. Our results show that RLHF improved the LLM's co-speech generative capabilities, producing more expressive, relevant and fluid movements.

**arXiv ID:** 2606.18747
</details>

<details>
<summary><strong>Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards</strong> - Yingyu Shan, Yuhang Guo, Zihao Cheng, Zeming Liu, Xiangrong Zhu, Xinyi Wang, Jiashu Yao, Wei Lin, Hongru Wang, Heyan Huang - [[pdf]](https://arxiv.org/pdf/2606.18810)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) has driven substantial progress in training LLMs for reasoning tasks, but representative methods such as GRPO assign uniform credit across all tokens, wasting gradient on routine tokens while under-crediting pivotal reasoning steps. Existing token-level credit assignment methods require resources beyond the model's own rollouts. GRPO variants rely on process reward models or ground-truth answers. Knowledge distillation assigns credit through per-token divergence but requires external teachers (On-Policy Distillation) or privileged information (On-Policy Self Distillation). However, these dependencies limit applicability in the pure RLVR setting. We observe that conditioning the model on its own verified trajectories induces a measurable per-token KL divergence between the original and conditioned distributions, and prove that distilling from a self-teacher constructed by verified trajectories leads to infeasible weighted-average solutions when multiple verified trajectories exist. We propose SC-GRPO (Self-Conditioned GRPO), which uses KL divergence mentioned before as a multiplicative weight on GRPO gradients. Across five benchmarks spanning math, code, and agentic tasks, SC-GRPO consistently outperforms 8.1% over GRPO and 5.9% over DAPO with stronger OOD performance. Moreover, SC-GRPO achieves higher performance than OPD.

**arXiv ID:** 2606.18810
</details>

<details>
<summary><strong>Reinforcement Learning Foundation Models Should Already Be A Thing</strong> - Abdelrahman Zighem, Jill-Jênn Vie - [[pdf]](https://arxiv.org/pdf/2606.18812)</summary>

**Abstract:** Foundation models for language and vision are powered by internet-scale data, while structured domains (tabular prediction, time-series forecasting, graph learning, reinforcement learning) are not. The substitute is synthetic data, which shifts the burden from collection to prior design. Such priors already exist for many structured tasks: TabPFN and its successors solve tabular classification with a transformer pretrained on a synthetic Bayesian prior.
We make two points. \textbf{First}, reinforcement learning is the conspicuous gap: sampling a synthetic MDP is as feasible as sampling a synthetic tabular dataset, yet no in-context RL work treats prior design as a primary objective. \textbf{Second}, MDPs admit a fixed-size sufficient statistic, independent of the episodes observed and tabular in shape, which makes them directly amenable to the attention-based architectures used for tabular foundation models, with a policy head replacing the supervised target. Together these define the agenda for an RL foundation model.
As a proof of concept, we train one model entirely on synthetic MDPs and show that, with no task-specific tuning, it solves held-out tabular benchmarks in context, both online and offline: online, in far fewer episodes than UCB-VI and tabular Q-learning, and offline, competitively with VI-LCB.

**arXiv ID:** 2606.18812
</details>

<details>
<summary><strong>UBP2: Uncertainty-Balanced Preference Planning for Efficient Preference-based Reinforcement Learning</strong> - Mohamed Nabail, Leo Cheng, Jingmin Wang, Nicholas Rhinehart - [[pdf]](https://arxiv.org/pdf/2606.19328)</summary>

**Abstract:** Preference-based RL provides an approach to learning reward models from pairwise comparisons of behaviors, bypassing the need for explicit reward design. However, existing methods typically rely on passive data collection and suffer from poor sample efficiency, especially during the early stages of learning. We introduce a model-based approach that actively directs exploration by jointly reasoning over uncertainties in the reward, dynamics, and value functions. Our method, Uncertainty-Balanced Preference Planning (UBP2), uses ensembles of reward, dynamics, and value function models to evaluate candidate trajectories according to a unified score that combines expected reward, terminal value, and epistemic uncertainty. Planning under this objective yields an explicit tradeoff between exploitation and information acquisition without requiring ad hoc exploration heuristics. Under standard regularity assumptions, we establish sublinear regret guarantees for both finite-horizon and infinite-horizon settings. Empirically, experiments on the Meta-World benchmark show UBP2 achieves substantially higher sample efficiency than model-free preference-based methods and non-optimistic model-based baselines.

**arXiv ID:** 2606.19328
</details>

<details>
<summary><strong>Retell, Reward, Repeat: Reinforcement Learning for Narrative Theory-Informed Story Retelling</strong> - David Y. Liu, Xanthe Muston, Dipankar Srirag, Aditya Joshi, Sebastian Sequoiah-Grayson - [[pdf]](https://arxiv.org/pdf/2601.17226)</summary>

**Abstract:** Counterfactual story retelling exposes LLM shortcomings in constrained narrative solution spaces where they can no longer rely on recalling memorised training data. Ground-truth-based post-training, such as SFT, fails to teach LLMs how to generate logical and rational narrative events. In this paper, we introduce Retell, Reward, Repeat (RRR), an RL-based pipeline synthesising Structuralist Narratology with scalar narrativity to teach storytelling structure. We extend the TimeTravel dataset with human-annotated stages of narrative equilibrium to evaluate reward models. By using d-RLAIF, RRR derives training signals from the narrativity of textual features without the need for reference outputs. Evaluations demonstrate that RRR-trained LLMs outperform few-shot and SFT baselines in logic, rationality, and completeness, with output quality additionally validated by blind human preference. Relying on a small, query-only dataset, RRR provides a linguistically grounded, cost-effective post-training mechanism for storytelling--a domain currently lacking effective post-training methods. RRR highlights the continued relevance of integrating established linguistic theories into contemporary NLP.

**arXiv ID:** 2601.17226
</details>

<details>
<summary><strong>WebSP-Eval: Evaluating Web Agents on Website Security and Privacy Tasks</strong> - Guruprasad Viswanathan Ramesh, Asmit Nayak, Basieem Siddique, Kassem Fawaz - [[pdf]](https://arxiv.org/pdf/2604.06367)</summary>

**Abstract:** Web agents automate browser tasks, ranging from simple form completion to complex workflows like ordering groceries. While current benchmarks evaluate general-purpose performance~(e.g., WebArena) or safety against malicious actions~(e.g., SafeArena), no existing framework assesses an agent's ability to successfully execute user-facing website security and privacy tasks, such as managing cookie preferences, configuring privacy-sensitive account settings, or revoking inactive sessions.
To address this gap, we introduce WebSP-Eval, an evaluation framework for measuring web agent performance on website security and privacy tasks. WebSP-Eval comprises 1) a manually crafted task dataset of 200 task instances across 28 websites; 2) a robust agentic system supporting account and initial state management across runs using a custom Google Chrome extension; and 3) an automated evaluator. We evaluate a total of 8 web agent instantiations using state-of-the-art multimodal large language models, conducting a fine-grained analysis across websites, task categories, and UI elements. Our evaluation reveals that current models suffer from limited autonomous exploration capabilities to reliably solve website security and privacy tasks, and struggle with specific task categories and websites. Crucially, we identify stateful UI elements are a primary reason for agent failure, with toggles causing more than 45% task failure across many models.

**arXiv ID:** 2604.06367
</details>

<details>
<summary><strong>Agents Trusting Agents? Restoring Lost Capabilities with Inclusive Healthcare</strong> - Alba Aguilera, Georgina Curto, Nardine Osman, Ahmed Al-Awah - [[pdf]](https://arxiv.org/pdf/2507.23644)</summary>

**Abstract:** Agent-based simulations have an untapped potential to inform social policies on urgent human development challenges in a non-invasive way, before these are implemented in real-world populations. This paper responds to the request from non-profit and governmental organizations to evaluate policies under discussion to improve equity in health care services for people experiencing homelessness (PEH) in the city of Barcelona. With this goal, we integrate the conceptual framework of the capability approach (CA), which is explicitly designed to promote and assess human well-being, to model and evaluate the behaviour of agents who represent PEH and social workers. We define a reinforcement learning environment where agents aim to restore their central human capabilities, under existing environmental and legal constraints. We use Bayesian inverse reinforcement learning (IRL) to calibrate profile-dependent behavioural parameters in PEH agents, modeling the degree of trust and engagement with social workers, which is reportedly a key element for the success of the policies in scope. Our results open a path to mitigate health inequity by building relationships of trust between social service workers and PEH.

**arXiv ID:** 2507.23644
</details>

<details>
<summary><strong>Beyond Monolingual Deep Research: Evaluating Agents and Retrievers with Cross-Lingual BrowseComp-Plus</strong> - Yuheng Lu, Qingcheng Zeng, Heli Qi, Puxuan Yu, Fuheng Zhao, Rui Yang, Hitomi Yanaka, Naoto Yokoya, Weihao Xuan - [[pdf]](https://arxiv.org/pdf/2606.15345)</summary>

**Abstract:** Deep research agents are increasingly evaluated on their ability to search for evidence, reason over retrieved sources, and produce grounded answers. Existing browsing benchmarks, however, largely assume that the user's query and the supporting evidence are written in the same language, leaving open whether agentic search systems can operate when relevant evidence appears in another language. We introduce XBCP (Cross-lingual BrowseComp-Plus), a controlled benchmark that preserves the English question-and-answer space of BrowseComp-Plus but varies the languages of the supporting documents. XBCP instantiates two complementary settings: in the cross-lingual setting, each query is paired with evidence in a single assigned language. In the multilingual setting, the full evidence corpus is distributed equally and randomly across 12 languages spanning high-resource and low-resource regimes. We evaluate four deep research agents using sparse and dense multilingual retrievers, measuring answer accuracy, evidence recall, search behavior, calibration, citation fidelity, and oracle retrieval. Results reveal substantial degradation when evidence is translated. Even strong, dense retrievers lose evidence recall, and agents become less calibrated and cite evidence less reliably. Notably, accuracy remains lower even when all gold evidence is supplied directly. These findings suggest that cross-lingual deep research exposes both retrieval failures and an independent, agent-side difficulty in integrating language-mismatched evidence.

**arXiv ID:** 2606.15345
</details>

<details>
<summary><strong>Quantum Annealing Enhanced Reinforcement Learning for Accurate Remaining Useful Lifetime Prediction</strong> - Manoranjan Gandhudi, Arunkumar V., G. R. Anil, Gangadharan G. R - [[pdf]](https://arxiv.org/pdf/2606.18503)</summary>

**Abstract:** Remaining useful life (RUL) estimation is central to predictive maintenance, where an unplanned failure can cost far more than the asset itself. Statistical degradation models miss the strong nonlinearity of real systems, and data-driven models often converge to suboptimal solutions in high-dimensional, non-convex search spaces. We propose a Quantum Annealing enhanced Q-Learning (QAQL) framework that couples the sampling behaviour of quantum annealing with the sequential decision making of Q-learning. Each Q-value update is encoded as a small quadratic unconstrained binary optimization (QUBO) whose ground state is the greedy action; rather than acting as a deterministic optimizer, the annealer returns a distribution over near-optimal actions across many reads, and this stochastic action selection supplies the exploration that curbs premature convergence on nonlinear degradation trajectories. The QUBO is solved on the D-Wave Advantage system using minor embedding, with the annealer woven into the reinforcement-learning loop rather than bolted on after training. We validate QAQL on two public benchmarks: the NASA C-MAPSS turbofan engine datasets and a device-fleet predictive maintenance dataset. Averaged over many independent runs and across six error metrics, QAQL outperforms the classical and quantum baselines considered in this study, with statistically significant improvements. The results indicate that quantum annealing is a usable, not merely theoretical, optimizer inside a reinforcement-learning loop for industrial predictive-maintenance applications.

**arXiv ID:** 2606.18503
</details>

<details>
<summary><strong>ToolChain-CRC: Conformal Risk Control for Agentic AI Under Retrieval and Tool-Use Drift</strong> - Jeffery Opoku, David Banahene - [[pdf]](https://arxiv.org/pdf/2606.18467)</summary>

**Abstract:** Modern AI agents retrieve documents, call tools, check intermediate information, and then produce a final answer or action. This creates a risk-control problem that is not visible from the final answer alone. A final response may look acceptable even when the retrieval was weak, a tool output was wrong, or an earlier step was unsupported. We propose ToolChain-CRC, a conformal risk-control method for retrieval-augmented and tool-using agents under drift. The method treats each agent run as a full trajectory of actions, observations, and final output. It builds step-level risk scores, combines them into a trajectory risk score, calibrates an accept-or-intervene rule, and adds an anytime alarm that can stop risky runs before the final answer. We prove trajectory-level risk control under exchangeable calibration runs, give a drift-aware extension with auditable constants, and prove an anytime escalation rule through a supermartingale construction. Experiments cover synthetic tool-chain drift, RAG/tool-use stress tests, public SQuAD-derived retrieval tasks, an API-free agentic QA case study, ablations, target-risk sensitivity checks, 20-seed robustness checks, a drift-margin audit, and a live RAG/tool-use agent benchmark. Across these settings, final-answer-only calibration can miss retrieval and tool failures, while trajectory-level calibration keeps accepted-trajectory risk below the target.

**arXiv ID:** 2606.18467
</details>

<details>
<summary><strong>When Does Trajectory-Level Supervision Permit Efficient Offline Reinforcement Learning?</strong> - Xuanfei Ren, Tengyang Xie - [[pdf]](https://arxiv.org/pdf/2606.18531)</summary>

**Abstract:** Offline reinforcement learning is typically analyzed under process-level reward supervision, yet many sequential decision datasets
record only trajectory-level outcomes. We develop a statistical theory for offline policy optimization from such outcome-level
supervision. We first study the canonical setting where the target remains the expected cumulative reward, but each offline trajectory
provides only a scalar label whose conditional mean is the cumulative return. We propose OPAC, a pessimistic actor-critic algorithm
that learns a latent reward model and optimizes a policy from trajectory-level labels. We prove a high-probability guarantee of order
$\widetilde O(H^2\sqrt{C_{sa}(\pi^\star)/n})$ and a matching lower bound, characterizing the sharp statistical cost of replacing
process-level rewards with one trajectory-level label. We then extend the principle to preference-based feedback, preserving the
leading horizon and concentrability dependence up to preference-model constants. Finally, we study generalized outcome-based offline
RL, where both the supervision and the objective are trajectory-level quantities induced by a nonlinear aggregation of latent per-step
rewards. This problem is not learnable in general: for all-success objectives, any offline learner may require $\Omega(2^H)$
trajectories even with deterministic transitions and constant concentrability. We then identify a tractable regime through two
structural coefficients, $\kappa_\mu(\sigma)$ and $\chi_\mu(\sigma)$, capturing information loss in outcome aggregation and
generalized Bellman updates, under which generalized OPAC achieves polynomial sample complexity. Together, our results delineate when
outcome-level supervision enables sample-efficient offline control and when missing process-level rewards create fundamental
statistical barriers.

**arXiv ID:** 2606.18531
</details>

<details>
<summary><strong>Model-Free Reinforcement Learning Control for Resilient Cyber-Physical Systems</strong> - Hugo O. Garcés, Alejandro J. Rojas, Bernardo A. Hernández, Andrés Escalona, Jonathan M. Palma, Md. Rezwan Parvez, Bhushan Gopaluni, Sirish L. Shah - [[pdf]](https://arxiv.org/pdf/2606.19069)</summary>

**Abstract:** This paper compares the performance of model-free controllers on a nonlinear system under cyberattacks, including false data injection and denial-of-service attacks. Four RL reward types are analyzed for accuracy, cost, and resilience. Results show that the Lyapunov reward offers the best resilience with low tracking error. Exponential mode also provides good trade-offs with acceptable resilience under moderate training conditions. Progressive and linear rewards converge faster but are less robust. RL-MPCs show strong steady-state resilience but require longer training times; RL-PID controllers are faster with significantly less training time. Proximal Policy Optimization outperforms Deep Deterministic Policy Gradient with a significant reduction in KPI variance. This study serves to highlight how well-designed RL rewards can improve performance and resilience against cyber threats.

**arXiv ID:** 2606.19069
</details>

<details>
<summary><strong>Reinforcement Learning for Accelerated Aerodynamic Shape Optimisation</strong> - Florian Sobieczky, Alfredo Lopez, Erika Dudkin, Christopher Lackner, Matthias Hochsteger, Bernhard Scheichl, Helmut Sobieczky - [[pdf]](https://arxiv.org/pdf/2507.17786)</summary>

**Abstract:** We introduce a reinforcement learning (RL) based adaptive optimization algorithm for aerodynamic shape optimization focused on dimensionality reduction. The form in which RL is applied here is that of a surrogate-based, actor-critic policy evaluation MCMC approach allowing for temporal 'freezing' of some of the parameters to be optimized. The goals are to minimize computational effort, and to use the observed optimization results for interpretation of the discovered extrema in terms of their role in achieving the desired flow-field.
By a sequence of local optimized parameter changes around intermediate CFD simulations acting as ground truth, it is possible to speed up the global optimization if (a) the local neighbourhoods of the parameters in which the changed parameters must reside are sufficiently large to compete with the grid-sized steps and its large number of simulations, and (b) the estimates of the rewards and costs on these neighbourhoods necessary for a good step-wise parameter adaption are sufficiently accurate. We give an example of a simple fluid-dynamical problem on which the method allows interpretation in the sense of a feature importance scoring.

**arXiv ID:** 2507.17786
</details>

<details>
<summary><strong>SRL: Combining SLIP Model and Reinforcement Learning for Agile Robotic Jumping</strong> - Xiaowen Hu, Linqi Ye, Yudi Zhu, Chenyue Shao, Rankun Li, Qingdu Li, Yan Peng - [[pdf]](https://arxiv.org/pdf/2606.18625)</summary>

**Abstract:** Robotic jumping is pivotal in applications such as search and rescue and logistics, where crossing obstacles and enhancing mobility efficiency are critical. The Spring-Loaded Inverted Pendulum (SLIP) model leverages simplified spring-mass dynamics that naturally encode biologically plausible hopping motions, yet its performance degrades on irregular terrain due to idealized assumptions regarding contact and joint dynamics. Meanwhile, Reinforcement Learning (RL) can adapt to diverse and complex environments but often requires extensive data from unguided exploration. The complementary strengths of SLIP's physically grounded baseline and RL's adaptive capabilities motivate a hybrid framework that overcomes these individual limitations. We therefore propose Spring-loaded Reinforcement Learning (SRL), which integrates SLIP-based feedforward control signals with RL-driven real-time feedback, enabling continuous optimization of robotic jumping. Experimental results demonstrate that SRL can achieve more stable jumps with much less training time than the baseline method, maintaining an average position tracking error below 0.1 m and velocity tracking errors within +/-3% of the target values. Through bipedal and quadrupedal simulations of ground and stair jumping, as well as sim-to-sim and sim-to-real validations, SRL exhibits robust adaptability to various task requirements and environmental complexities, underscoring its potential for real-world deployment.

**arXiv ID:** 2606.18625
</details>

<details>
<summary><strong>A Scalable Embodied Intelligence Platform for Seamless Real-to-Sim-to-Real Transfer of Household Mobile Manipulation Tasks</strong> - Kui Yang, Xianlei Long, Haoxuan Li, Yan Ding, Chao Chen - [[pdf]](https://arxiv.org/pdf/2606.18646)</summary>

**Abstract:** Mobile manipulation is a fundamental capability in embodied intelligence robotics. The growing demand for robust and generalizable manipulation in unstructured household environments has driven rapid progress in embodied intelligence platforms. However, achieving a seamless transfer across the real-to-sim-to-real cycle faces three key challenges, including costly high-fidelity simulation scenes reconstruction, the complexity of systematic strategy evaluation in simulation, and incompatible real-world deployments. To address these challenges, we develop BestMan, a scalable and seamless real-to-sim-to-real platform that bridges the gap between the simulation and the real world, enabling effective strategy development, integration, and deployment for household mobile manipulation. Specifically, we design a novel Automated Scene Generation (ASG) module to reconstruct realistic simulations from real observations. Then, we propose a simulation-guided task formalization and skill learning architecture that supports the flexible integration and large-scale evaluations of hybrid skill strategies in simulation. Finally, to enhance the real-world scalability, we develop a Hardware-agnostic and Unified Middleware (HUM) to ensure seamless and compatible sim-to-real transfer across heterogeneous mobile manipulators for real deployments. Experimental results demonstrate the superior performance of our proposed platform in establishing standardized benchmarks and facilitating promising research in the field of mobile manipulation.

**arXiv ID:** 2606.18646
</details>

<details>
<summary><strong>Human-AI Agent Interaction in a Business Context</strong> - Kathrin Paimann, Elizangela Valarini, Sebastian Juhl - [[pdf]](https://arxiv.org/pdf/2606.18716)</summary>

**Abstract:** As AI agents are increasingly integrated into core business processes, understanding and designing effective interaction patterns between humans and AI agents becomes crucial for value creation. This study identifies and evaluates principles and criteria for a positive User Experience (UX) with AI agents, along with methods for its measurement. We identify user expectations and needs to facilitate adoption, build trust, and support user-centered decision-making by development teams. Using a mixed-methods approach that combines qualitative and quantitative techniques, we explore interaction patterns between humans and AI agents. The findings from this exploratory research serve as the basis to develop a survey experiment which evaluates the effectiveness of specific design elements on a larger scale. This foundational research contributes to the development of more intuitive and effective human-AI agent interactions in business settings.

**arXiv ID:** 2606.18716
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
