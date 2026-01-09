# Agent arXiv Daily

**Last Updated:** 2026-01-09 03:06:52

**Total Papers:** 99

## Table of Contents

- [Agent Applications](#agent-applications)
- [Benchmarks and Datasets](#benchmarks-and-datasets)
- [LLM Agents](#llm-agents)
- [Multi-Agent Systems](#multi-agent-systems)
- [Other Agent Research](#other-agent-research)
- [Reinforcement Learning](#reinforcement-learning)

<details open>
<summary><h2>Agent Applications (6 papers)</h2></summary>

<details>
<summary><strong>Attachment Styles and AI Chatbot Interactions Among College Students</strong> - Ziqi Lin, Taiyu Hou - [[pdf]](https://arxiv.org/pdf/2601.04217)</summary>

**Abstract:** The use of large language model (LLM)-based AI chatbots among college students has increased rapidly, yet little is known about how individual psychological attributes shape students' interaction patterns with these technologies. This qualitative study explored how college students with different attachment styles describe their interactions with ChatGPT. Using semi-structured interviews with seven undergraduate students and grounded theory analysis, we identified three main themes: (1) AI as a low-risk emotional space, where participants across attachment styles valued the non-judgmental and low-stakes nature of AI interactions; (2) attachment-congruent patterns of AI engagement, where securely attached students integrated AI as a supplementary tool within their existing support systems, while avoidantly attached students used AI to buffer vulnerability and maintain interpersonal boundaries; and (3) the paradox of AI intimacy, capturing the tension between students' willingness to disclose personal information to AI while simultaneously recognizing its limitations as a relational partner. These findings suggest that attachment orientations play an important role in shaping how students experience and interpret their interactions with AI chatbots, extending attachment theory to the domain of human-AI interaction.

**arXiv ID:** 2601.04217
</details>

<details>
<summary><strong>Can Consumer Chatbots Reason? A Student-Led Field Experiment Embedded in an "AI-for-All" Undergraduate Course</strong> - Amarda Shehu, Adonyas Ababu, Asma Akbary, Griffin Allen, Aroush Baig, Tereana Battle, Elias Beall, Christopher Byrom, Matt Dean, Kate Demarco, Ethan Douglass, Luis Granados, Layla Hantush, Andy Hay, Eleanor Hay, Caleb Jackson, Jaewon Jang, Carter Jones, Quanyang Li, Adrian Lopez, Logan Massimo, Garrett McMullin, Ariana Mendoza Maldonado, Eman Mirza, Hadiya Muddasar, Sara Nuwayhid, Brandon Pak, Ashley Petty, Dryden Rancourt, Lily Rodriguez, Corbin Rogers, Jacob Schiek, Taeseo Seok, Aarav Sethi, Giovanni Vitela, Winston Williams, Jagan Yetukuri - [[pdf]](https://arxiv.org/pdf/2601.04225)</summary>

**Abstract:** Claims about whether large language model (LLM) chatbots "reason" are typically debated using curated benchmarks and laboratory-style evaluation protocols. This paper offers a complementary perspective: a student-led field experiment embedded as a midterm project in UNIV 182 (AI4All) at George Mason University, a Mason Core course designed for undergraduates across disciplines with no expected prior STEM exposure. Student teams designed their own reasoning tasks, ran them on widely used consumer chatbots representative of current capabilities, and evaluated both (i) answer correctness and (ii) the validity of the chatbot's stated reasoning (for example, cases where an answer is correct but the explanation is not, or vice versa). Across eight teams that reported standardized scores, students contributed 80 original reasoning prompts spanning six categories: pattern completion, transformation rules, spatial/visual reasoning, quantitative reasoning, relational/logic reasoning, and analogical reasoning. These prompts yielded 320 model responses plus follow-up explanations. Aggregating team-level results, OpenAI GPT-5 and Claude 4.5 achieved the highest mean answer accuracy (86.2% and 83.8%), followed by Grok 4 (82.5%) and Perplexity (73.1%); explanation validity showed a similar ordering (81.2%, 80.0%, 77.5%, 66.2%). Qualitatively, teams converged on a consistent error signature: strong performance on short, structured math and pattern items but reduced reliability on spatial/visual reasoning and multi-step transformations, with frequent "sound right but reason wrong" explanations. The assignment's primary contribution is pedagogical: it operationalizes AI literacy as experimental practice (prompt design, measurement, rater disagreement, and interpretability/grounding) while producing a reusable, student-generated corpus of reasoning probes grounded in authentic end-user interaction.

**arXiv ID:** 2601.04225
</details>

<details>
<summary><strong>AI Agents as Policymakers in Simulated Epidemics</strong> - Goshi Aoki, Navid Ghaffarzadegan - [[pdf]](https://arxiv.org/pdf/2601.04245)</summary>

**Abstract:** AI agents are increasingly deployed as quasi-autonomous systems for specialized tasks, yet their potential as computational models of decision-making remains underexplored. We develop a generative AI agent to study repetitive policy decisions during an epidemic, embedding the agent, prompted to act as a city mayor, within a simulated SEIR environment. Each week, the agent receives updated epidemiological information, evaluates the evolving situation, and sets business restriction levels. The agent is equipped with a dynamic memory that weights past events by recency and is evaluated in both single- and ensemble-agent settings across environments of varying complexity. Across scenarios, the agent exhibits human-like reactive behavior, tightening restrictions in response to rising cases and relaxing them as risk declines. Crucially, providing the agent with brief systems-level knowledge of epidemic dynamics, highlighting feedbacks between disease spread and behavioral responses, substantially improves decision quality and stability. The results illustrate how theory-informed prompting can shape emergent policy behavior in AI agents. These findings demonstrate that generative AI agents, when situated in structured environments and guided by minimal domain theory, can serve as powerful computational models for studying decision-making and policy design in complex social systems.

**arXiv ID:** 2601.04245
</details>

<details>
<summary><strong>Gavel: Agent Meets Checklist for Evaluating LLMs on Long-Context Legal Summarization</strong> - Yao Dou, Wei Xu - [[pdf]](https://arxiv.org/pdf/2601.04424)</summary>

**Abstract:** Large language models (LLMs) now support contexts of up to 1M tokens, but their effectiveness on complex long-context tasks remains unclear. In this paper, we study multi-document legal case summarization, where a single case often spans many documents totaling 100K-500K tokens. We introduce Gavel-Ref, a reference-based evaluation framework with multi-value checklist evaluation over 26 items, as well as residual fact and writing-style evaluations. Using Gavel-Ref, we go beyond the single aggregate scores reported in prior work and systematically evaluate 12 frontier LLMs on 100 legal cases ranging from 32K to 512K tokens, primarily from 2025. Our results show that even the strongest model, Gemini 2.5 Pro, achieves only around 50 of $S_{\text{Gavel-Ref}}$, highlighting the difficulty of the task. Models perform well on simple checklist items (e.g., filing date) but struggle on multi-value or rare ones such as settlements and monitor reports. As LLMs continue to improve and may surpass human-written summaries -- making human references less reliable -- we develop Gavel-Agent, an efficient and autonomous agent scaffold that equips LLMs with six tools to navigate and extract checklists directly from case documents. With Qwen3, Gavel-Agent reduces token usage by 36% while resulting in only a 7% drop in $S_{\text{checklist}}$ compared to end-to-end extraction with GPT-4.1.

**arXiv ID:** 2601.04424
</details>

<details>
<summary><strong>VCB Bench: An Evaluation Benchmark for Audio-Grounded Large Language Model Conversational Agents</strong> - Jiliang Hu, Wenfu Wang, Zuchao Li, Chenxing Li, Yiyang Zhao, Hanzhao Li, Liqiang Zhang, Meng Yu, Dong Yu - [[pdf]](https://arxiv.org/pdf/2510.11098)</summary>

**Abstract:** Recent advances in large audio language models (LALMs) have greatly enhanced multimodal conversational systems. However, existing benchmarks remain limited -- they are mainly English-centric, rely on synthetic speech, and lack comprehensive, discriminative evaluation across multiple dimensions. To address these gaps, we present Voice Chat Bot Bench (VCB Bench) -- a high-quality Chinese benchmark built entirely on real human speech. VCB Bench evaluates LALMs from three complementary perspectives: instruction following (including speech-level control beyond text commands), knowledge understanding (general knowledge, reasoning, and daily dialogue), and robustness (stability under perturbations in content, environment, and speaker traits). Experiments on representative LALMs reveal notable performance gaps and highlight future directions for improvement. VCB Bench provides a reproducible and fine-grained evaluation framework, offering standardized methodology and practical insights for advancing Chinese voice conversational models.

**arXiv ID:** 2510.11098
</details>

<details>
<summary><strong>Forging Spatial Intelligence: A Roadmap of Multi-Modal Data Pre-Training for Autonomous Systems</strong> - Song Wang, Lingdong Kong, Xiaolu Liu, Hao Shi, Wentong Li, Jianke Zhu, Steven C. H. Hoi - [[pdf]](https://arxiv.org/pdf/2512.24385)</summary>

**Abstract:** The rapid advancement of autonomous systems, including self-driving vehicles and drones, has intensified the need to forge true Spatial Intelligence from multi-modal onboard sensor data. While foundation models excel in single-modal contexts, integrating their capabilities across diverse sensors like cameras and LiDAR to create a unified understanding remains a formidable challenge. This paper presents a comprehensive framework for multi-modal pre-training, identifying the core set of techniques driving progress toward this goal. We dissect the interplay between foundational sensor characteristics and learning strategies, evaluating the role of platform-specific datasets in enabling these advancements. Our central contribution is the formulation of a unified taxonomy for pre-training paradigms: ranging from single-modality baselines to sophisticated unified frameworks that learn holistic representations for advanced tasks like 3D object detection and semantic occupancy prediction. Furthermore, we investigate the integration of textual inputs and occupancy representations to facilitate open-world perception and planning. Finally, we identify critical bottlenecks, such as computational efficiency and model scalability, and propose a roadmap toward general-purpose multi-modal foundation models capable of achieving robust Spatial Intelligence for real-world deployment.

**arXiv ID:** 2512.24385
</details>

</details>

<details open>
<summary><h2>Benchmarks and Datasets (15 papers)</h2></summary>

<details>
<summary><strong>SAGE-32B: Agentic Reasoning via Iterative Distillation</strong> - Basab Jha, Firoj Paudel, Ujjwal Puri, Ethan Henkel, Zhang Yuting, Mateusz Kowalczyk, Mei Huang, Choi Donghyuk, Wang Junhao - [[pdf]](https://arxiv.org/pdf/2601.04237)</summary>

**Abstract:** We demonstrate SAGE-32B, a 32 billion parameter language model that focuses on agentic reasoning and long range planning tasks. Unlike chat models that aim for general conversation fluency, SAGE-32B is designed to operate in an agentic loop, emphasizing task decomposition, tool usage, and error recovery. The model is initialized from the Qwen2.5-32B pretrained model and fine tuned using Iterative Distillation, a two stage training process that improves reasoning performance through rigorously tested feedback loops. SAGE-32B also introduces an inverse reasoning approach, which uses a meta cognition head to forecast potential failures in the planning process before execution. On agentic reasoning benchmarks including MMLU-Pro, AgentBench, and MATH-500, SAGE-32B achieves higher success rates in multi tool usage scenarios compared to similarly sized baseline models, while remaining competitive on standard reasoning evaluations. Model weights are publicly released at this https URL

**arXiv ID:** 2601.04237
</details>

<details>
<summary><strong>Actively Obtaining Environmental Feedback for Autonomous Action Evaluation Without Predefined Measurements</strong> - Hong Su - [[pdf]](https://arxiv.org/pdf/2601.04235)</summary>

**Abstract:** Obtaining reliable feedback from the environment is a fundamental capability for intelligent agents to evaluate the correctness of their actions and to accumulate reusable knowledge. However, most existing approaches rely on predefined measurements or fixed reward signals, which limits their applicability in open-ended and dynamic environments where new actions may require previously unknown forms of feedback. To address these limitations, this paper proposes an Actively Feedback Getting model, in which an AI agent proactively interacts with the environment to discover, screen, and verify feedback without relying on predefined measurements. Rather than assuming explicit feedback definitions, the proposed method exploits action-induced environmental differences to identify target feedback that is not specified in advance, based on the observation that actions inevitably produce measurable changes in the environment. In addition, a self-triggering mechanism, driven by internal objectives such as improved accuracy, precision, and efficiency, is introduced to autonomously plan and adjust actions, thereby enabling faster and more focused feedback acquisition without external commands. Experimental results demonstrate that the proposed active approach significantly improves the efficiency and robustness of factor identification.

**arXiv ID:** 2601.04235
</details>

<details>
<summary><strong>Adversarial Yet Cooperative: Multi-Perspective Reasoning in Retrieved-Augmented Language Models</strong> - Can Xu, Lingyong Yan, Jiayi Wu, Haosen Wang, Shuaiqiang Wang, Yuchen Li, Jizhou Huang, Dawei Yin, Xiang Li - [[pdf]](https://arxiv.org/pdf/2601.04651)</summary>

**Abstract:** Recent advances in synergizing large reasoning models (LRMs) with retrieval-augmented generation (RAG) have shown promising results, yet two critical challenges remain: (1) reasoning models typically operate from a single, unchallenged perspective, limiting their ability to conduct deep, self-correcting reasoning over external documents, and (2) existing training paradigms rely excessively on outcome-oriented rewards, which provide insufficient signal for shaping the complex, multi-step reasoning process. To address these issues, we propose an Reasoner-Verifier framework named Adversarial Reasoning RAG (ARR). The Reasoner and Verifier engage in reasoning on retrieved evidence and critiquing each other's logic while being guided by process-aware advantage that requires no external scoring model. This reward combines explicit observational signals with internal model uncertainty to jointly optimize reasoning fidelity and verification rigor. Experiments on multiple benchmarks demonstrate the effectiveness of our method.

**arXiv ID:** 2601.04651
</details>

<details>
<summary><strong>ParaCodex: A Profiling-Guided Autonomous Coding Agent for Reliable Parallel Code Generation and Translation</strong> - Erel Kaplan, Tomer Bitan, Lian Ghrayeb, Le Chen, Tom Yotam, Niranjan Hasabnis, Gal Oren - [[pdf]](https://arxiv.org/pdf/2601.04327)</summary>

**Abstract:** Parallel programming is central to HPC and AI, but producing code that is correct and fast remains challenging, especially for OpenMP GPU offload, where data movement and tuning dominate. Autonomous coding agents can compile, test, and profile on target hardware, but outputs are brittle without domain scaffolding.
We present ParaCodex, an HPC-engineer workflow that turns a Codex-based agent into an autonomous OpenMP GPU offload system using staged hotspot analysis, explicit data planning, correctness gating, and profiling-guided refinement. We evaluate translation from serial CPU kernels to OpenMP GPU offload kernels on HeCBench, Rodinia, and NAS. After excluding five kernels, ParaCodex succeeded on all 31 valid kernels. The generated kernels improved GPU time over reference OpenMP implementations in 25/31 cases, achieving geometric-mean speedups of 3x on HeCBench and 5x on Rodinia, and outperforming a zero-shot Codex baseline on all suites. We also evaluate CUDA to OpenMP offload translation on ParEval, where ParaCodex maintains high compilation and validation rates in code-only and end-to-end settings.

**arXiv ID:** 2601.04327
</details>

<details>
<summary><strong>Improving and Evaluating Open Deep Research Agents</strong> - Doaa Allabadi, Kyle Bradbury, Jordan M. Malof - [[pdf]](https://arxiv.org/pdf/2508.10152)</summary>

**Abstract:** We focus here on Deep Research Agents (DRAs), which are systems that can take a natural language prompt from a user, and then autonomously search for, and utilize, internet-based content to address the prompt. Recent DRAs have demonstrated impressive capabilities on public benchmarks however, recent research largely involves proprietary closed-source systems. At the time of this work, we only found one open-source DRA, termed Open Deep Research (ODR). In this work we adapt the challenging recent BrowseComp benchmark to compare ODR to existing proprietary systems. We propose BrowseComp-Small (BC-Small), comprising a subset of BrowseComp, as a more computationally-tractable DRA benchmark for academic labs. We benchmark ODR and two other proprietary systems on BC-Small: one system from Anthropic and one system from Google. We find that all three systems achieve 0% accuracy on the test set of 60 questions. We introduce three strategic improvements to ODR, resulting in the ODR+ model, which achieves a state-of-the-art 10% success rate on BC-Small among both closed-source and open-source systems. We report ablation studies indicating that all three of our improvements contributed to the success of ODR+.

**arXiv ID:** 2508.10152
</details>

<details>
<summary><strong>Human-in-the-Loop Testing of AI Agents for Air Traffic Control with a Regulated Assessment Framework</strong> - Ben Carvell, Marc Thomas, Andrew Pace, Christopher Dorney, George De Ath, Richard Everson, Nick Pepper, Adam Keane, Samuel Tomlinson, Richard Cannon - [[pdf]](https://arxiv.org/pdf/2601.04288)</summary>

**Abstract:** We present a rigorous, human-in-the-loop evaluation framework for assessing the performance of AI agents on the task of Air Traffic Control, grounded in a regulator-certified simulator-based curriculum used for training and testing real-world trainee controllers. By leveraging legally regulated assessments and involving expert human instructors in the evaluation process, our framework enables a more authentic and domain-accurate measurement of AI performance. This work addresses a critical gap in the existing literature: the frequent misalignment between academic representations of Air Traffic Control and the complexities of the actual operational environment. It also lays the foundations for effective future human-machine teaming paradigms by aligning machine performance with human assessment targets.

**arXiv ID:** 2601.04288
</details>

<details>
<summary><strong>Agent+P: Guiding UI Agents via Symbolic Planning</strong> - Shang Ma, Xusheng Xiao, Yanfang Ye - [[pdf]](https://arxiv.org/pdf/2510.06042)</summary>

**Abstract:** Large Language Model (LLM)-based UI agents show great promise for UI automation but often hallucinate in long-horizon tasks due to their lack of understanding of the global UI transition structure. To address this, we introduce AGENT+P, a novel framework that leverages symbolic planning to guide LLM-based UI agents. Specifically, we model an app's UI transition structure as a UI Transition Graph (UTG), which allows us to reformulate the UI automation task as a pathfinding problem on the UTG. This further enables an off-the-shelf symbolic planner to generate a provably correct and optimal high-level plan, preventing the agent from redundant exploration and guiding the agent to achieve the automation goals. AGENT+P is designed as a plug-and-play framework to enhance existing UI agents. Evaluation on the AndroidWorld benchmark demonstrates that AGENT+P improves the success rates of state-of-the-art UI agents by up to 14.31% and reduces the action steps by 37.70%.

**arXiv ID:** 2510.06042
</details>

<details>
<summary><strong>See, Explain, and Intervene: A Few-Shot Multimodal Agent Framework for Hateful Meme Moderation</strong> - Naquee Rizwan, Subhankar Swain, Paramananda Bhaskar, Gagan Aryan, Shehryaar Shah Khan, Animesh Mukherjee - [[pdf]](https://arxiv.org/pdf/2601.04692)</summary>

**Abstract:** In this work, we examine hateful memes from three complementary angles - how to detect them, how to explain their content and how to intervene them prior to being posted - by applying a range of strategies built on top of generative AI models. To the best of our knowledge, explanation and intervention have typically been studied separately from detection, which does not reflect real-world conditions. Further, since curating large annotated datasets for meme moderation is prohibitively expensive, we propose a novel framework that leverages task-specific generative multimodal agents and the few-shot adaptability of large multimodal models to cater to different types of memes. We believe this is the first work focused on generalizable hateful meme moderation under limited data conditions, and has strong potential for deployment in real-world production scenarios. Warning: Contains potentially toxic contents.

**arXiv ID:** 2601.04692
</details>

<details>
<summary><strong>Fame Fades, Nature Remains: Disentangling the Character Identity of Role-Playing Agents</strong> - Yonghyun Jun, Junhyuk Choi, Jihyeong Park, Hwanhee Lee - [[pdf]](https://arxiv.org/pdf/2601.04716)</summary>

**Abstract:** Despite the rapid proliferation of Role-Playing Agents (RPAs) based on Large Language Models (LLMs), the structural dimensions defining a character's identity remain weakly formalized, often treating characters as arbitrary text inputs. In this paper, we propose the concept of \textbf{Character Identity}, a multidimensional construct that disentangles a character into two distinct layers: \textbf{(1) Parametric Identity}, referring to character-specific knowledge encoded from the LLM's pre-training, and \textbf{(2) Attributive Identity}, capturing fine-grained behavioral properties such as personality traits and moral values. To systematically investigate these layers, we construct a unified character profile schema and generate both Famous and Synthetic characters under identical structural constraints. Our evaluation across single-turn and multi-turn interactions reveals two critical phenomena. First, we identify \textit{"Fame Fades"}: while famous characters hold a significant advantage in initial turns due to parametric knowledge, this edge rapidly vanishes as models prioritize accumulating conversational context over pre-trained priors. Second, we find that \textit{"Nature Remains"}: while models robustly portray general personality traits regardless of polarity, RPA performance is highly sensitive to the valence of morality and interpersonal relationships. Our findings pinpoint negative social natures as the primary bottleneck in RPA fidelity, guiding future character construction and evaluation.

**arXiv ID:** 2601.04716
</details>

<details>
<summary><strong>DocDancer: Towards Agentic Document-Grounded Information Seeking</strong> - Qintong Zhang, Xinjie Lv, Jialong Wu, Baixuan Li, Zhengwei Tao, Guochen Yan, Huanyao Zhang, Bin Wang, Jiahao Xu, Haitao Mi, Wentao Zhang - [[pdf]](https://arxiv.org/pdf/2601.05163)</summary>

**Abstract:** Document Question Answering (DocQA) focuses on answering questions grounded in given documents, yet existing DocQA agents lack effective tool utilization and largely rely on closed-source models. In this work, we introduce DocDancer, an end-to-end trained open-source Doc agent. We formulate DocQA as an information-seeking problem and propose a tool-driven agent framework that explicitly models document exploration and comprehension. To enable end-to-end training of such agents, we introduce an Exploration-then-Synthesis data synthesis pipeline that addresses the scarcity of high-quality training data for DocQA. Training on the synthesized data, the trained models on two long-context document understanding benchmarks, MMLongBench-Doc and DocBench, show their effectiveness. Further analysis provides valuable insights for the agentic tool design and synthetic data.

**arXiv ID:** 2601.05163
</details>

<details>
<summary><strong>NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents</strong> - Jingzhe Ding, Shengda Long, Changxin Pu, Huan Zhou, Hongwan Gao, Xiang Gao, Chao He, Yue Hou, Fei Hu, Zhaojian Li, Weiran Shi, Zaiyuan Wang, Daoguang Zan, Chenchen Zhang, Xiaoxu Zhang, Qizhi Chen, Xianfu Cheng, Bo Deng, Qingshui Gu, Kai Hua, Juntao Lin, Pai Liu, Mingchen Li, Xuanguang Pan, Zifan Peng, Yujia Qin, Yong Shan, Zhewen Tan, Weihao Xie, Zihan Wang, Yishuo Yuan, Jiayu Zhang, Enduo Zhao, Yunfei Zhao, He Zhu, Liya Zhu, Chenyang Zou, Ming Ding, Jianpeng Jiao, Jiaheng Liu, Minghao Liu, Qian Liu, Chongyang Tao, Jian Yang, Tong Yang, Zhaoxiang Zhang, Xinjie Chen, Wenhao Huang, Ge Zhang - [[pdf]](https://arxiv.org/pdf/2512.12730)</summary>

**Abstract:** Recent advances in coding agents suggest rapid progress toward autonomous software development, yet existing benchmarks fail to rigorously evaluate the long-horizon capabilities required to build complete software systems. Most prior evaluations focus on localized code generation, scaffolded completion, or short-term repair tasks, leaving open the question of whether agents can sustain coherent reasoning, planning, and execution over the extended horizons demanded by real-world repository construction. To address this gap, we present NL2Repo Bench, a benchmark explicitly designed to evaluate the long-horizon repository generation ability of coding agents. Given only a single natural-language requirements document and an empty workspace, agents must autonomously design the architecture, manage dependencies, implement multi-module logic, and produce a fully installable Python library. Our experiments across state-of-the-art open- and closed-source models reveal that long-horizon repository generation remains largely unsolved: even the strongest agents achieve below 40% average test pass rates and rarely complete an entire repository correctly. Detailed analysis uncovers fundamental long-horizon failure modes, including premature termination, loss of global coherence, fragile cross-file dependencies, and inadequate planning over hundreds of interaction steps. NL2Repo Bench establishes a rigorous, verifiable testbed for measuring sustained agentic competence and highlights long-horizon reasoning as a central bottleneck for the next generation of autonomous coding agents.

**arXiv ID:** 2512.12730
</details>

<details>
<summary><strong>Agent-Dice: Disentangling Knowledge Updates via Geometric Consensus for Agent Continual Learning</strong> - Zheng Wu, Xingyu Lou, Xinbei Ma, Yansi Li, Weiwen Liu, Weinan Zhang, Jun Wang, Zhuosheng Zhang - [[pdf]](https://arxiv.org/pdf/2601.03641)</summary>

**Abstract:** Large Language Model (LLM)-based agents significantly extend the utility of LLMs by interacting with dynamic environments. However, enabling agents to continually learn new tasks without catastrophic forgetting remains a critical challenge, known as the stability-plasticity dilemma. In this work, we argue that this dilemma fundamentally arises from the failure to explicitly distinguish between common knowledge shared across tasks and conflicting knowledge introduced by task-specific interference. To address this, we propose Agent-Dice, a parameter fusion framework based on directional consensus evaluation. Concretely, Agent-Dice disentangles knowledge updates through a two-stage process: geometric consensus filtering to prune conflicting gradients, and curvature-based importance weighting to amplify shared semantics. We provide a rigorous theoretical analysis that establishes the validity of the proposed fusion scheme and offers insight into the origins of the stability-plasticity dilemma. Extensive experiments on GUI agents and tool-use agent domains demonstrate that Agent-Dice exhibits outstanding continual learning performance with minimal computational overhead and parameter updates. The codes are available at this https URL.

**arXiv ID:** 2601.03641
</details>

<details>
<summary><strong>Phasor Agents: Oscillatory Graphs with Three-Factor Plasticity and Sleep-Staged Learning</strong> - Rodja Trappe - [[pdf]](https://arxiv.org/pdf/2601.04362)</summary>

**Abstract:** Phasor Agents are dynamical systems whose internal state is a Phasor Graph: a weighted graph of coupled Stuart-Landau oscillators. A Stuart-Landau oscillator is a minimal stable "rhythm generator" (the normal form near a Hopf bifurcation); each oscillator is treated as an abstract computational unit (inspired by, but not claiming to model, biological oscillatory populations). In this interpretation, oscillator phase tracks relative timing (coherence), while amplitude tracks local gain or activity. Relative phase structure serves as a representational medium; coupling weights are learned via three-factor local plasticity - eligibility traces gated by sparse global modulators and oscillation-timed write windows - without backpropagation.
A central challenge in oscillatory substrates is stability: online weight updates can drive the network into unwanted regimes (e.g., global synchrony), collapsing representational diversity. We therefore separate wake tagging from offline consolidation, inspired by synaptic tagging-and-capture and sleep-stage dynamics: deep-sleep-like gated capture commits tagged changes safely, while REM-like replay reconstructs and perturbs experience for planning.
A staged experiment suite validates each mechanism with ablations and falsifiers: eligibility traces preserve credit under delayed modulation; compression-progress signals pass timestamp-shuffle controls; phase-coherent retrieval reaches 4x diffusive baselines under noise; wake/sleep separation expands stable learning by 67 percent under matched weight-norm budgets; REM replay improves maze success rate by +45.5 percentage points; and a Tolman-style latent-learning signature - immediate competence and detour advantage after unrewarded exploration, consistent with an internal model - emerges from replay (Tolman, 1948).
The codebase and all artifacts are open-source.

**arXiv ID:** 2601.04362
</details>

<details>
<summary><strong>Toward Maturity-Based Certification of Embodied AI: Quantifying Trustworthiness Through Measurement Mechanisms</strong> - Michael C. Darling, Alan H. Hesu, Michael A. Mardikes, Brian C. McGuigan, Reed M. Milewicz - [[pdf]](https://arxiv.org/pdf/2601.03470)</summary>

**Abstract:** We propose a maturity-based framework for certifying embodied AI systems through explicit measurement mechanisms. We argue that certifiable embodied AI requires structured assessment frameworks, quantitative scoring mechanisms, and methods for navigating multi-objective trade-offs inherent in trustworthiness evaluation. We demonstrate this approach using uncertainty quantification as an exemplar measurement mechanism and illustrate feasibility through an Uncrewed Aircraft System (UAS) detection case study.

**arXiv ID:** 2601.03470
</details>

<details>
<summary><strong>ImagineNav++: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination</strong> - Teng Wang, Xinxin Zhao, Wenzhe Cai, Changyin Sun - [[pdf]](https://arxiv.org/pdf/2512.17435)</summary>

**Abstract:** Visual navigation is a fundamental capability for autonomous home-assistance robots, enabling long-horizon tasks such as object search. While recent methods have leveraged Large Language Models (LLMs) to incorporate commonsense reasoning and improve exploration efficiency, their planning remains constrained by textual representations, which cannot adequately capture spatial occupancy or scene geometry--critical factors for navigation decisions. We explore whether Vision-Language Models (VLMs) can achieve mapless visual navigation using only onboard RGB/RGB-D streams, unlocking their potential for spatial perception and planning. We achieve this through an imagination-powered navigation framework, ImagineNav++, which imagines future observation images from candidate robot views and translates navigation planning into a simple best-view image selection problem for VLMs. First, a future-view imagination module distills human navigation preferences to generate semantically meaningful viewpoints with high exploration potential. These imagined views then serve as visual prompts for the VLM to identify the most informative viewpoint. To maintain spatial consistency, we develop a selective foveation memory mechanism, which hierarchically integrates keyframe observations via a sparse-to-dense framework, constructing a compact yet comprehensive memory for long-term spatial reasoning. This approach transforms goal-oriented navigation into a series of tractable point-goal navigation tasks. Extensive experiments on open-vocabulary object and instance navigation benchmarks show that ImagineNav++ achieves SOTA performance in mapless settings, even surpassing most map-based methods, highlighting the importance of scene imagination and memory in VLM-based spatial reasoning.

**arXiv ID:** 2512.17435
</details>

</details>

<details open>
<summary><h2>LLM Agents (7 papers)</h2></summary>

<details>
<summary><strong>XGrammar 2: Dynamic and Efficient Structured Generation Engine for Agentic LLMs</strong> - Linzhang Li, Yixin Dong, Guanjie Wang, Ziyi Xu, Alexander Jiang, Tianqi Chen - [[pdf]](https://arxiv.org/pdf/2601.04426)</summary>

**Abstract:** Modern LLM agents are required to handle increasingly complex structured generation tasks, such as tool calling and conditional structured generation. These tasks are significantly more dynamic than predefined structures, posing new challenges to the current structured generation engines. In this paper, we propose XGrammar 2, a highly optimized structured generation engine for agentic LLMs. XGrammar 2 accelerates the mask generation for these dynamic structured generation tasks through a new dynamic dispatching semantics: TagDispatch. We further introduce a just-in-time (JIT) compilation method to reduce compilation time and a cross-grammar caching mechanism to leverage the common sub-structures across different grammars. Additionally, we extend the previous PDA-based mask generation algorithm to the Earley-parser-based one and design a repetition compression algorithm to handle repetition structures in grammars. Evaluation results show that XGrammar 2 can achieve more than 6x speedup over the existing structured generation engines. Integrated with an LLM inference engine, XGrammar 2 can handle dynamic structured generation tasks with near-zero overhead.

**arXiv ID:** 2601.04426
</details>

<details>
<summary><strong>BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents</strong> - Yunhao Feng, Yige Li, Yutao Wu, Yingshui Tan, Yanming Guo, Yifan Ding, Kun Zhai, Xingjun Ma, Yugang Jiang - [[pdf]](https://arxiv.org/pdf/2601.04566)</summary>

**Abstract:** Large language model (LLM) agents execute tasks through multi-step workflows that combine planning, memory, and tool use. While this design enables autonomy, it also expands the attack surface for backdoor threats. Backdoor triggers injected into specific stages of an agent workflow can persist through multiple intermediate states and adversely influence downstream outputs. However, existing studies remain fragmented and typically analyze individual attack vectors in isolation, leaving the cross-stage interaction and propagation of backdoor triggers poorly understood from an agent-centric perspective. To fill this gap, we propose \textbf{BackdoorAgent}, a modular and stage-aware framework that provides a unified, agent-centric view of backdoor threats in LLM agents. BackdoorAgent structures the attack surface into three functional stages of agentic workflows, including \textbf{planning attacks}, \textbf{memory attacks}, and \textbf{tool-use attacks}, and instruments agent execution to enable systematic analysis of trigger activation and propagation across different stages. Building on this framework, we construct a standardized benchmark spanning four representative agent applications: \textbf{Agent QA}, \textbf{Agent Code}, \textbf{Agent Web}, and \textbf{Agent Drive}, covering both language-only and multimodal settings. Our empirical analysis shows that \textit{triggers implanted at a single stage can persist across multiple steps and propagate through intermediate states.} For instance, when using a GPT-based backbone, we observe trigger persistence in 43.58\% of planning attacks, 77.97\% of memory attacks, and 60.28\% of tool-stage attacks, highlighting the vulnerabilities of the agentic workflow itself to backdoor threats. To facilitate reproducibility and future research, our code and benchmark are publicly available at GitHub.

**arXiv ID:** 2601.04566
</details>

<details>
<summary><strong>AgentDevel: Reframing Self-Evolving LLM Agents as Release Engineering</strong> - Di Zhang - [[pdf]](https://arxiv.org/pdf/2601.04620)</summary>

**Abstract:** Recent progress in large language model (LLM) agents has largely focused on embedding self-improvement mechanisms inside the agent or searching over many concurrent variants. While these approaches can raise aggregate scores, they often yield unstable and hard-to-audit improvement trajectories, making it difficult to guarantee non-regression or to reason about failures across versions. We reframe agent improvement as \textbf{release engineering}: agents are treated as shippable artifacts, and improvement is externalized into a regression-aware release pipeline. We introduce \textbf{AgentDevel}, a release engineering pipeline that iteratively runs the current agent, produces implementation-blind, symptom-level quality signals from execution traces, synthesizes a single release candidate (RC) via executable diagnosis, and promotes it under flip-centered gating. AgentDevel features three core designs: (i) an implementation-blind LLM critic that characterizes failure appearances without accessing agent internals, (ii) script-based executable diagnosis that aggregates dominant symptom patterns and produces auditable engineering specifications, and (iii) flip-centered gating that prioritizes pass to fail regressions and fail to pass fixes as first-class evidence. Unlike population-based search or in-agent self-refinement, AgentDevel maintains a single canonical version line and emphasizes non-regression as a primary objective. Experiments on execution-heavy benchmarks demonstrate that AgentDevel yields stable improvements with significantly fewer regressions while producing reproducible, auditable artifacts. Overall, AgentDevel provides a practical development discipline for building, debugging, and releasing LLM agents as software development.

**arXiv ID:** 2601.04620
</details>

<details>
<summary><strong>AT$^2$PO: Agentic Turn-based Policy Optimization via Tree Search</strong> - Zefang Zong, Dingwei Chen, Yang Li, Qi Yi, Bo Zhou, Chengming Li, Bo Qian, Peng Chen, Jie Jiang - [[pdf]](https://arxiv.org/pdf/2601.04767)</summary>

**Abstract:** LLM agents have emerged as powerful systems for tackling multi-turn tasks by interleaving internal reasoning and external tool interactions. Agentic Reinforcement Learning has recently drawn significant research attention as a critical post-training paradigm to further refine these capabilities. In this paper, we present AT$^2$PO (Agentic Turn-based Policy Optimization via Tree Search), a unified framework for multi-turn agentic RL that addresses three core challenges: limited exploration diversity, sparse credit assignment, and misaligned policy optimization. AT$^2$PO introduces a turn-level tree structure that jointly enables Entropy-Guided Tree Expansion for strategic exploration and Turn-wise Credit Assignment for fine-grained reward propagation from sparse outcomes. Complementing this, we propose Agentic Turn-based Policy Optimization, a turn-level learning objective that aligns policy updates with the natural decision granularity of agentic interactions. ATPO is orthogonal to tree search and can be readily integrated into any multi-turn RL pipeline. Experiments across seven benchmarks demonstrate consistent improvements over the state-of-the-art baseline by up to 1.84 percentage points in average, with ablation studies validating the effectiveness of each component. Our code is available at this https URL.

**arXiv ID:** 2601.04767
</details>

<details>
<summary><strong>MineNPC-Task: Task Suite for Memory-Aware Minecraft Agents</strong> - Tamil Sudaravan Mohan Doss, Michael Xu, Sudha Rao, Andrew D. Wilson, Balasaravanan Thoravi Kumaravel - [[pdf]](https://arxiv.org/pdf/2601.05215)</summary>

**Abstract:** We present \textsc{MineNPC-Task}, a user-authored benchmark and evaluation harness for testing memory-aware, mixed-initiative LLM agents in open-world \emph{Minecraft}. Rather than relying on synthetic prompts, tasks are elicited from formative and summative co-play with expert players, normalized into parametric templates with explicit preconditions and dependency structure, and paired with machine-checkable validators under a bounded-knowledge policy that forbids out-of-world shortcuts. The harness captures plan/act/memory events-including plan previews, targeted clarifications, memory reads and writes, precondition checks, and repair attempts and reports outcomes relative to the total number of attempted subtasks, derived from in-world evidence.
As an initial snapshot, we instantiate the framework with GPT-4o and evaluate \textbf{216} subtasks across \textbf{8} experienced players. We observe recurring breakdown patterns in code execution, inventory/tool handling, referencing, and navigation, alongside recoveries supported by mixed-initiative clarifications and lightweight memory. Participants rated interaction quality and interface usability positively, while highlighting the need for stronger memory persistence across tasks. We release the complete task suite, validators, logs, and harness to support transparent, reproducible evaluation of future memory-aware embodied agents.

**arXiv ID:** 2601.05215
</details>

<details>
<summary><strong>Beyond Static Summarization: Proactive Memory Extraction for LLM Agents</strong> - Chengyuan Yang, Zequn Sun, Wei Wei, Wei Hu - [[pdf]](https://arxiv.org/pdf/2601.04463)</summary>

**Abstract:** Memory management is vital for LLM agents to handle long-term interaction and personalization. Most research focuses on how to organize and use memory summary, but often overlooks the initial memory extraction stage. In this paper, we argue that existing summary-based methods have two major limitations based on the recurrent processing theory. First, summarization is "ahead-of-time", acting as a blind "feed-forward" process that misses important details because it doesn't know future tasks. Second, extraction is usually "one-off", lacking a feedback loop to verify facts, which leads to the accumulation of information loss. To address these issues, we propose proactive memory extraction (namely ProMem). Unlike static summarization, ProMem treats extraction as an iterative cognitive process. We introduce a recurrent feedback loop where the agent uses self-questioning to actively probe the dialogue history. This mechanism allows the agent to recover missing information and correct errors. Our ProMem significantly improves the completeness of the extracted memory and QA accuracy. It also achieves a superior trade-off between extraction quality and token cost.

**arXiv ID:** 2601.04463
</details>

<details>
<summary><strong>Cutting AI Research Costs: How Task-Aware Compression Makes Large Language Model Agents Affordable</strong> - Zuhair Ahmed Khan Taha, Mohammed Mudassir Uddin, Shahnawaz Alam - [[pdf]](https://arxiv.org/pdf/2601.05191)</summary>

**Abstract:** When researchers deploy large language models for autonomous tasks like reviewing literature or generating hypotheses, the computational bills add up quickly. A single research session using a 70-billion parameter model can cost around $127 in cloud fees, putting these tools out of reach for many academic labs. We developed AgentCompress to tackle this problem head-on. The core idea came from a simple observation during our own work: writing a novel hypothesis clearly demands more from the model than reformatting a bibliography. Why should both tasks run at full precision? Our system uses a small neural network to gauge how hard each incoming task will be, based only on its opening words, then routes it to a suitably compressed model variant. The decision happens in under a millisecond. Testing across 500 research workflows in four scientific fields, we cut compute costs by 68.3% while keeping 96.2% of the original success rate. For labs watching their budgets, this could mean the difference between running experiments and sitting on the sidelines

**arXiv ID:** 2601.05191
</details>

</details>

<details open>
<summary><h2>Multi-Agent Systems (26 papers)</h2></summary>

<details>
<summary><strong>A Closed-Loop Multi-Agent System Driven by LLMs for Meal-Level Personalized Nutrition Management</strong> - Muqing Xu - [[pdf]](https://arxiv.org/pdf/2601.04491)</summary>

**Abstract:** Personalized nutrition management aims to tailor dietary guidance to an individual's intake and phenotype, but most existing systems handle food logging, nutrient analysis and recommendation separately. We present a next-generation mobile nutrition assistant that combines image based meal logging with an LLM driven multi agent controller to provide meal level closed loop support. The system coordinates vision, dialogue and state management agents to estimate nutrients from photos and update a daily intake budget. It then adapts the next meal plan to user preferences and dietary constraints. Experiments with SNAPMe meal images and simulated users show competitive nutrient estimation, personalized menus and efficient task plans. These findings demonstrate the feasibility of multi agent LLM control for personalized nutrition and reveal open challenges in micronutrient estimation from images and in large scale real world studies.

**arXiv ID:** 2601.04491
</details>

<details>
<summary><strong>GUITester: Enabling GUI Agents for Exploratory Defect Discovery</strong> - Yifei Gao, Jiang Wu, Xiaoyi Chen, Yifan Yang, Zhe Cui, Tianyi Ma, Jiaming Zhang, Jitao Sang - [[pdf]](https://arxiv.org/pdf/2601.04500)</summary>

**Abstract:** Exploratory GUI testing is essential for software quality but suffers from high manual costs. While Multi-modal Large Language Model (MLLM) agents excel in navigation, they fail to autonomously discover defects due to two core challenges: \textit{Goal-Oriented Masking}, where agents prioritize task completion over reporting anomalies, and \textit{Execution-Bias Attribution}, where system defects are misidentified as agent errors. To address these, we first introduce \textbf{GUITestBench}, the first interactive benchmark for this task, featuring 143 tasks across 26 defects. We then propose \textbf{GUITester}, a multi-agent framework that decouples navigation from verification via two modules: (i) a \textit{Planning-Execution Module (PEM)} that proactively probes for defects via embedded testing intents, and (ii) a \textit{Hierarchical Reflection Module (HRM)} that resolves attribution ambiguity through interaction history analysis. GUITester achieves an F1-score of 48.90\% (Pass@3) on GUITestBench, outperforming state-of-the-art baselines (33.35\%). Our work demonstrates the feasibility of autonomous exploratory testing and provides a robust foundation for future GUI quality assurance~\footnote{Our code is now available in~\href{this https URL}{this https URL}}.

**arXiv ID:** 2601.04500
</details>

<details>
<summary><strong>CircuitLM: A Multi-Agent LLM-Aided Design Framework for Generating Circuit Schematics from Natural Language Prompts</strong> - Khandakar Shakib Al Hasan, Syed Rifat Raiyan, Hasin Mahtab Alvee, Wahid Sadik - [[pdf]](https://arxiv.org/pdf/2601.04505)</summary>

**Abstract:** Generating accurate circuit schematics from high-level natural language descriptions remains a persistent challenge in electronics design, as large language models (LLMs) frequently hallucinate in granular details, violate electrical constraints, and produce non-machine-readable outputs. We present CircuitLM, a novel multi-agent LLM-aided circuit design pipeline that translates user prompts into structured, visually interpretable CircuitJSON schematics through five sequential stages: (i) LLM-based component identification, (ii) canonical pinout retrieval, (iii) chain-of-thought reasoning by an electronics expert agent, (iv) JSON schematic synthesis, and (v) force-directed SVG visualization. Anchored by a curated, embedding-powered component knowledge base. While LLMs often violate electrical constraints, CircuitLM bridges this gap by grounding generation in a verified and dynamically extensible component database, initially comprising 50 components. To ensure safety, we incorporate a hybrid evaluation framework, namely Dual-Metric Circuit Validation (DMCV), validated against human-expert assessments, which achieves high fidelity in microcontroller-centric designs. We evaluate the system on 100 diverse embedded-systems prompts across six LLMs and introduce DMCV to assess both structural and electrical validity. This work bridges natural language input to deployable hardware designs, enabling reliable circuit prototyping by non-experts. Our code and data will be made public upon acceptance.

**arXiv ID:** 2601.04505
</details>

<details>
<summary><strong>TCAndon-Router: Adaptive Reasoning Router for Multi-Agent Collaboration</strong> - Jiuzhou Zhao, Chunrong Chen, Chenqi Qiao, Lebin Zheng, Minqi Han, Yanchi Liu Yongzhou Xu Xiaochuan Xu Min Zhang - [[pdf]](https://arxiv.org/pdf/2601.04544)</summary>

**Abstract:** Multi-Agent Systems(MAS) have become a powerful paradigm for building high performance intelligent applications. Within these systems, the router responsible for determining which expert agents should handle a given query plays a crucial role in overall performance. Existing routing strategies generally fall into two categories: performance routing, which balances latency and cost across models of different sizes, and task routing, which assigns queries to domain-specific experts to improve accuracy. In real-world enterprise applications, task routing is more suitable; however, most existing approaches rely on static single-label decisions, which introduce two major limitations: (i) difficulty in seamlessly integrating new agents as business domains expand, and (ii) routing conflicts caused by overlapping agent capabilities, ultimately degrading accuracy and this http URL address these challenges, we propose TCAndon-Router(TCAR): an adaptive reasoning router for multi-agent collaboration. Unlike traditional routers, TCAR supports dynamic agent onboarding and first generates a natural-language reasoning chain before predicting a set of candidate agents capable of handling the query. In addition, we design a collaborative execution pipeline in which selected agents independently produce responses, which are then aggregated and refined into a single high-quality response by a dedicated Refining this http URL on public datasets and real enterprise data demonstrate that TCAR significantly improves routing accuracy, reduces routing conflicts, and remains robust in ambiguous scenarios. We have released TCAR at this https URL to support future research on explainable and collaborative multi-agent routing.

**arXiv ID:** 2601.04544
</details>

<details>
<summary><strong>Autonomous Agents on Blockchains: Standards, Execution Models, and Trust Boundaries</strong> - Saad Alqithami - [[pdf]](https://arxiv.org/pdf/2601.04583)</summary>

**Abstract:** Advances in large language models have enabled agentic AI systems that can reason, plan, and interact with external tools to execute multi-step workflows, while public blockchains have evolved into a programmable substrate for value transfer, access control, and verifiable state transitions. Their convergence introduces a high-stakes systems challenge: designing standard, interoperable, and secure interfaces that allow agents to observe on-chain state, formulate transaction intents, and authorize execution without exposing users, protocols, or organizations to unacceptable security, governance, or economic risks. This survey systematizes the emerging landscape of agent-blockchain interoperability through a systematic literature review, identifying 317 relevant works from an initial pool of over 3000 records. We contribute a five-part taxonomy of integration patterns spanning read-only analytics, simulation and intent generation, delegated execution, autonomous signing, and multi-agent workflows; a threat model tailored to agent-driven transaction pipelines that captures risks ranging from prompt injection and policy misuse to key compromise, adversarial execution dynamics, and multi-agent collusion; and a comparative capability matrix analyzing more than 20 representative systems across 13 dimensions, including custody models, permissioning, policy enforcement, observability, and recovery. Building on the gaps revealed by this analysis, we outline a research roadmap centered on two interface abstractions: a Transaction Intent Schema for portable and unambiguous goal specification, and a Policy Decision Record for auditable, verifiable policy enforcement across execution environments. We conclude by proposing a reproducible evaluation suite and benchmarks for assessing the safety, reliability, and economic robustness of agent-mediated on-chain execution.

**arXiv ID:** 2601.04583
</details>

<details>
<summary><strong>ResMAS: Resilience Optimization in LLM-based Multi-agent Systems</strong> - Zhilun Zhou, Zihan Liu, Jiahe Liu, Qingyu Shao, Yihan Wang, Kun Shao, Depeng Jin, Fengli Xu - [[pdf]](https://arxiv.org/pdf/2601.04694)</summary>

**Abstract:** Large Language Model-based Multi-Agent Systems (LLM-based MAS), where multiple LLM agents collaborate to solve complex tasks, have shown impressive performance in many areas. However, MAS are typically distributed across different devices or environments, making them vulnerable to perturbations such as agent failures. While existing works have studied the adversarial attacks and corresponding defense strategies, they mainly focus on reactively detecting and mitigating attacks after they occur rather than proactively designing inherently resilient systems. In this work, we study the resilience of LLM-based MAS under perturbations and find that both the communication topology and prompt design significantly influence system resilience. Motivated by these findings, we propose ResMAS: a two-stage framework for enhancing MAS resilience. First, we train a reward model to predict the MAS's resilience, based on which we train a topology generator to automatically design resilient topology for specific tasks through reinforcement learning. Second, we introduce a topology-aware prompt optimization method that refines each agent's prompt based on its connections and interactions with other agents. Extensive experiments across a range of tasks show that our approach substantially improves MAS resilience under various constraints. Moreover, our framework demonstrates strong generalization ability to new tasks and models, highlighting its potential for building resilient MASs.

**arXiv ID:** 2601.04694
</details>

<details>
<summary><strong>Beyond Monolithic Architectures: A Multi-Agent Search and Knowledge Optimization Framework for Agentic Search</strong> - Yiqun Chen, Lingyong Yan, Zixuan Yang, Erhan Zhang, Jiashu Zhao, Shuaiqiang Wang, Dawei Yin, Jiaxin Mao - [[pdf]](https://arxiv.org/pdf/2601.04703)</summary>

**Abstract:** Agentic search has emerged as a promising paradigm for complex information seeking by enabling Large Language Models (LLMs) to interleave reasoning with tool use. However, prevailing systems rely on monolithic agents that suffer from structural bottlenecks, including unconstrained reasoning outputs that inflate trajectories, sparse outcome-level rewards that complicate credit assignment, and stochastic search noise that destabilizes learning. To address these challenges, we propose \textbf{M-ASK} (Multi-Agent Search and Knowledge), a framework that explicitly decouples agentic search into two complementary roles: Search Behavior Agents, which plan and execute search actions, and Knowledge Management Agents, which aggregate, filter, and maintain a compact internal context. This decomposition allows each agent to focus on a well-defined subtask and reduces interference between search and context construction. Furthermore, to enable stable coordination, M-ASK employs turn-level rewards to provide granular supervision for both search decisions and knowledge updates. Experiments on multi-hop QA benchmarks demonstrate that M-ASK outperforms strong baselines, achieving not only superior answer accuracy but also significantly more stable training dynamics.\footnote{The source code for M-ASK is available at this https URL.}

**arXiv ID:** 2601.04703
</details>

<details>
<summary><strong>When Single-Agent with Skills Replace Multi-Agent Systems and When They Fail</strong> - Xiaoxiao Li - [[pdf]](https://arxiv.org/pdf/2601.04748)</summary>

**Abstract:** Multi-agent AI systems have proven effective for complex reasoning. These systems are compounded by specialized agents, which collaborate through explicit communication, but incur substantial computational overhead. A natural question arises: can we achieve similar modularity benefits with a single agent that selects from a library of skills? We explore this question by viewing skills as internalized agent behaviors. From this perspective, a multi-agent system can be compiled into an equivalent single-agent system, trading inter-agent communication for skill selection. Our preliminary experiments suggest this approach can substantially reduce token usage and latency while maintaining competitive accuracy on reasoning benchmarks. However, this efficiency raises a deeper question that has received little attention: how does skill selection scale as libraries grow?
Drawing on principles from cognitive science, we propose that LLM skill selection exhibits bounded capacity analogous to human decision-making. We investigate the scaling behavior of skill selection and observe a striking pattern. Rather than degrading gradually, selection accuracy remains stable up to a critical library size, then drops sharply, indicating a phase transition reminiscent of capacity limits in human cognition. Furthermore, we find evidence that semantic confusability among similar skills, rather than library size alone, plays a central role in this degradation. This perspective suggests that hierarchical organization, which has long helped humans manage complex choices, may similarly benefit AI systems. Our initial results with hierarchical routing support this hypothesis. This work opens new questions about the fundamental limits of semantic-based skill selection in LLMs and offers a cognitive-grounded framework and practical guidelines for designing scalable skill-based agents.

**arXiv ID:** 2601.04748
</details>

<details>
<summary><strong>Orchestrating Intelligence: Confidence-Aware Routing for Efficient Multi-Agent Collaboration across Multi-Scale Models</strong> - Jingbo Wang, Sendong Zhao, Jiatong Liu, Haochun Wang, Wanting Li, Bing Qin, Ting Liu - [[pdf]](https://arxiv.org/pdf/2601.04861)</summary>

**Abstract:** While multi-agent systems (MAS) have demonstrated superior performance over single-agent approaches in complex reasoning tasks, they often suffer from significant computational inefficiencies. Existing frameworks typically deploy large language models (LLMs) uniformly across all agent roles, failing to account for the varying cognitive demands of different reasoning stages. We address this inefficiency by proposing OI-MAS framework, a novel multi-agent framework that implements an adaptive model-selection policy across a heterogeneous pool of multi-scale LLMs. Specifically, OI-MAS introduces a state-dependent routing mechanism that dynamically selects agent roles and model scales throughout the reasoning process. In addition, we introduce a confidence-aware mechanism that selects appropriate model scales conditioned on task complexity, thus reducing unnecessary reliance on large-scale models. Experimental results show that OI-MAS consistently outperforms baseline multi-agent systems, improving accuracy by up to 12.88\% while reducing cost by up to 79.78\%.

**arXiv ID:** 2601.04861
</details>

<details>
<summary><strong>Precomputing Multi-Agent Path Replanning using Temporal Flexibility: A Case Study on the Dutch Railway Network</strong> - Issa Hanou, Eric Kemmeren, Devin Wild Thomas, Mathijs de Weerdt - [[pdf]](https://arxiv.org/pdf/2601.04884)</summary>

**Abstract:** Executing a multi-agent plan can be challenging when an agent is delayed, because this typically creates conflicts with other agents. So, we need to quickly find a new safe plan. Replanning only the delayed agent often does not result in an efficient plan, and sometimes cannot even yield a feasible plan. On the other hand, replanning other agents may lead to a cascade of changes and delays. We show how to efficiently replan by tracking and using the temporal flexibility of other agents while avoiding cascading delays. This flexibility is the maximum delay an agent can take without changing the order of or further delaying more agents. Our algorithm, FlexSIPP, precomputes all possible plans for the delayed agent, also returning the changes for the other agents, for any single-agent delay within the given scenario. We demonstrate our method in a real-world case study of replanning trains in the densely-used Dutch railway network. Our experiments show that FlexSIPP provides effective solutions, relevant to real-world adjustments, and within a reasonable timeframe.

**arXiv ID:** 2601.04884
</details>

<details>
<summary><strong>AgentTutor: Empowering Personalized Learning with Multi-Turn Interactive Teaching in Intelligent Education Systems</strong> - Yuxin Liu, Zeqing Song, Jiong Lou, Chentao Wu, Jie Li - [[pdf]](https://arxiv.org/pdf/2601.04219)</summary>

**Abstract:** The rapid advancement of large-scale language models (LLMs) has shown their potential to transform intelligent education systems (IESs) through automated teaching and learning support applications. However, current IESs often rely on single-turn static question-answering, which fails to assess learners' cognitive levels, cannot adjust teaching strategies based on real-time feedback, and is limited to providing simple one-off responses. To address these issues, we introduce AgentTutor, a multi-turn interactive intelligent education system to empower personalized learning. It features an LLM-powered generative multi-agent system and a learner-specific personalized learning profile environment that dynamically optimizes and delivers teaching strategies based on learners' learning status, personalized goals, learning preferences, and multimodal study materials. It includes five key modules: curriculum decomposition, learner assessment, dynamic strategy, teaching reflection, and knowledge & experience memory. We conducted extensive experiments on multiple benchmark datasets, AgentTutor significantly enhances learners' performance while demonstrating strong effectiveness in multi-turn interactions and competitiveness in teaching quality among other baselines.

**arXiv ID:** 2601.04219
</details>

<details>
<summary><strong>Integrating Multi-Agent Simulation, Behavioral Forensics, and Trust-Aware Machine Learning for Adaptive Insider Threat Detection</strong> - Firdous Kausar, Asmah Muallem, Naw Safrin Sattar, Mohamed Zakaria Kurdi - [[pdf]](https://arxiv.org/pdf/2601.04243)</summary>

**Abstract:** We present a hybrid framework for adaptive insider-threat detection that tightly integrates multi-agent simulation (MAS), layered Security Information and Event Management (SIEM) correlation, behavioral and communication forensics, trust-aware machine learning, and Theory-of-Mind (ToM) reasoning. Intelligent agents operate in a simulated enterprise environment, generating both behavioral events and cognitive intent signals that are ingested by a centralized SIEM. We evaluate four system variants: a Layered SIEM-Core (LSC) baseline, a Cognitive-Enriched SIEM (CE-SIEM) incorporating ToM and communication forensics, an Evidence-Gated SIEM (EG-SIEM) introducing precision-focused validation mechanisms, and an Enron-enabled EG-SIEM (EG-SIEM-Enron) that augments evidence gating with a pretrained email forensics module calibrated on Enron corpora. Across ten simulation runs involving eight malicious insiders, CE-SIEM achieves perfect recall (1.000) and improves actor-level F1 from 0.521 (LSC) to 0.774. EG-SIEM raises actor-level F1 to 0.922 and confirmed-alert precision to 0.997 while reducing false positives to 0.2 per run. EG-SIEM-Enron preserves high precision (1.000 confirmed-alert precision; 0.0 false positives per run), slightly improves actor-level F1 to 0.933, and reduces detection latency (average TTD 10.26 steps versus 15.20 for EG-SIEM). These results demonstrate that cognitive context improves sensitivity, evidence-gated validation enables high-precision, low-noise detection, and pretrained communication calibration can further accelerate high-confidence insider threat identification.

**arXiv ID:** 2601.04243
</details>

<details>
<summary><strong>3D-Agent:Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation</strong> - Jusheng Zhang, Yijia Fan, Zimo Wen, Jian Wang, Keze Wang - [[pdf]](https://arxiv.org/pdf/2601.04404)</summary>

**Abstract:** Driven by applications in autonomous driving robotics and augmented reality 3D object annotation presents challenges beyond 2D annotation including spatial complexity occlusion and viewpoint inconsistency Existing approaches based on single models often struggle to address these issues effectively We propose Tri MARF a novel framework that integrates tri modal inputs including 2D multi view images textual descriptions and 3D point clouds within a multi agent collaborative architecture to enhance large scale 3D annotation Tri MARF consists of three specialized agents a vision language model agent for generating multi view descriptions an information aggregation agent for selecting optimal descriptions and a gating agent that aligns textual semantics with 3D geometry for refined captioning Extensive experiments on Objaverse LVIS Objaverse XL and ABO demonstrate that Tri MARF substantially outperforms existing methods achieving a CLIPScore of 88 point 7 compared to prior state of the art methods retrieval accuracy of 45 point 2 and 43 point 8 on ViLT R at 5 and a throughput of up to 12000 objects per hour on a single NVIDIA A100 GPU

**arXiv ID:** 2601.04404
</details>

<details>
<summary><strong>Belief in Authority: Impact of Authority in Multi-Agent Evaluation Framework</strong> - Junhyuk Choi, Jeongyoun Kwon, Heeju Kim, Haeun Cho, Hayeong Jung, Sehee Min, Bugeun Kim - [[pdf]](https://arxiv.org/pdf/2601.04790)</summary>

**Abstract:** Multi-agent systems utilizing large language models often assign authoritative roles to improve performance, yet the impact of authority bias on agent interactions remains underexplored. We present the first systematic analysis of role-based authority bias in free-form multi-agent evaluation using ChatEval. Applying French and Raven's power-based theory, we classify authoritative roles into legitimate, referent, and expert types and analyze their influence across 12-turn conversations. Experiments with GPT-4o and DeepSeek R1 reveal that Expert and Referent power roles exert stronger influence than Legitimate power roles. Crucially, authority bias emerges not through active conformity by general agents, but through authoritative roles consistently maintaining their positions while general agents demonstrate flexibility. Furthermore, authority influence requires clear position statements, as neutral responses fail to generate bias. These findings provide key insights for designing multi-agent frameworks with asymmetric interaction patterns.

**arXiv ID:** 2601.04790
</details>

<details>
<summary><strong>From Idea to Co-Creation: A Planner-Actor-Critic Framework for Agent Augmented 3D Modeling</strong> - Jin Gao, Saichandu Juluri - [[pdf]](https://arxiv.org/pdf/2601.05016)</summary>

**Abstract:** We present a framework that extends the Actor-Critic architecture to creative 3D modeling through multi-agent self-reflection and human-in-the-loop supervision. While existing approaches rely on single-prompt agents that directly execute modeling commands via tools like Blender MCP, our approach introduces a Planner-Actor-Critic architecture. In this design, the Planner coordinates modeling steps, the Actor executes them, and the Critic provides iterative feedback, while human users act as supervisors and advisors throughout the process. Through systematic comparison between single-prompt modeling and our reflective multi-agent approach, we demonstrate improvements in geometric accuracy, aesthetic quality, and task completion rates across diverse 3D modeling scenarios. Our evaluation reveals that critic-guided reflection, combined with human supervisory input, reduces modeling errors and increases complexity and quality of the result compared to direct single-prompt execution. This work establishes that structured agent self-reflection, when augmented by human oversight and advisory guidance, produces higher-quality 3D models while maintaining efficient workflow integration through real-time Blender synchronization.

**arXiv ID:** 2601.05016
</details>

<details>
<summary><strong>Agent-as-a-Judge</strong> - Runyang You, Hongru Cai, Caiqi Zhang, Qiancheng Xu, Meng Liu, Tiezheng Yu, Yongqi Li, Wenjie Li - [[pdf]](https://arxiv.org/pdf/2601.05111)</summary>

**Abstract:** LLM-as-a-Judge has revolutionized AI evaluation by leveraging large language models for scalable assessments. However, as evaluands become increasingly complex, specialized, and multi-step, the reliability of LLM-as-a-Judge has become constrained by inherent biases, shallow single-pass reasoning, and the inability to verify assessments against real-world observations. This has catalyzed the transition to Agent-as-a-Judge, where agentic judges employ planning, tool-augmented verification, multi-agent collaboration, and persistent memory to enable more robust, verifiable, and nuanced evaluations. Despite the rapid proliferation of agentic evaluation systems, the field lacks a unified framework to navigate this shifting landscape. To bridge this gap, we present the first comprehensive survey tracing this evolution. Specifically, we identify key dimensions that characterize this paradigm shift and establish a developmental taxonomy. We organize core methodologies and survey applications across general and professional domains. Furthermore, we analyze frontier challenges and identify promising research directions, ultimately providing a clear roadmap for the next generation of agentic evaluation.

**arXiv ID:** 2601.05111
</details>

<details>
<summary><strong>Beyond Detection: Exploring Evidence-based Multi-Agent Debate for Misinformation Intervention and Persuasion</strong> - Chen Han, Yijia Ma, Jin Tan, Wenzhen Zheng, Xijin Tang - [[pdf]](https://arxiv.org/pdf/2511.07267)</summary>

**Abstract:** Multi-agent debate (MAD) frameworks have emerged as promising approaches for misinformation detection by simulating adversarial reasoning. While prior work has focused on detection accuracy, it overlooks the importance of helping users understand the reasoning behind factual judgments and develop future resilience. The debate transcripts generated during MAD offer a rich but underutilized resource for transparent reasoning. In this study, we introduce ED2D, an evidence-based MAD framework that extends previous approach by incorporating factual evidence retrieval. More importantly, ED2D is designed not only as a detection framework but also as a persuasive multi-agent system aimed at correcting user beliefs and discouraging misinformation sharing. We compare the persuasive effects of ED2D-generated debunking transcripts with those authored by human experts. Results demonstrate that ED2D outperforms existing baselines across three misinformation detection benchmarks. When ED2D generates correct predictions, its debunking transcripts exhibit persuasive effects comparable to those of human experts; However, when ED2D misclassifies, its accompanying explanations may inadvertently reinforce users'misconceptions, even when presented alongside accurate human explanations. Our findings highlight both the promise and the potential risks of deploying MAD systems for misinformation intervention. We further develop a public community website to help users explore ED2D, fostering transparency, critical thinking, and collaborative fact-checking.

**arXiv ID:** 2511.07267
</details>

<details>
<summary><strong>FinDeepForecast: A Live Multi-Agent System for Benchmarking Deep Research Agents in Financial Forecasting</strong> - Xiangyu Li, Xuan Yao, Guohao Qi, Fengbin Zhu, Kelvin J.L. Koa, Xiang Yao Ng, Ziyang Liu, Xingyu Ni, Chang Liu, Yonghui Yang, Yang Zhang, Wenjie Wang, Fuli Feng, Chao Wang, Huanbo Luan, Xiaofen Xing, Xiangmin Xu, Tat-Seng Chua, Ke-Wei Huang - [[pdf]](https://arxiv.org/pdf/2601.05039)</summary>

**Abstract:** Deep Research (DR) Agents powered by advanced Large Language Models (LLMs) have fundamentally shifted the paradigm for completing complex research tasks. Yet, a comprehensive and live evaluation of their forecasting performance on real-world, research-oriented tasks in high-stakes domains (e.g., finance) remains underexplored. We introduce FinDeepForecast, the first live, end-to-end multi-agent system for automatically evaluating DR agents by continuously generating research-oriented fi- nancial forecasting tasks. This system is equipped with a dual-track taxonomy, enabling the dynamic generation of recurrent and non-recurrent forecasting tasks at both corporate and macro levels. With this system, we generate FinDeepForecastBench, a weekly evaluation benchmark over a ten-week horizon, encompassing 8 global economies and 1,314 listed companies, and evaluate 13 representative methods. Extensive experiments show that, while DR agents consistently outperform strong baselines, their performance still falls short of genuine forward-looking financial reasoning. We expect the proposed FinDeepForecast system to consistently facilitate future advancements of DR agents in research-oriented financial forecasting tasks. The benchmark and leaderboard are publicly available on the OpenFinArena Platform.

**arXiv ID:** 2601.05039
</details>

<details>
<summary><strong>Transformer-based Multi-agent Reinforcement Learning for Separation Assurance in Structured and Unstructured Airspaces</strong> - Arsyi Aziz, Peng Wei - [[pdf]](https://arxiv.org/pdf/2601.04401)</summary>

**Abstract:** Conventional optimization-based metering depends on strict adherence to precomputed schedules, which limits the flexibility required for the stochastic operations of Advanced Air Mobility (AAM). In contrast, multi-agent reinforcement learning (MARL) offers a decentralized, adaptive framework that can better handle uncertainty, required for safe aircraft separation assurance. Despite this advantage, current MARL approaches often overfit to specific airspace structures, limiting their adaptability to new configurations. To improve generalization, we recast the MARL problem in a relative polar state space and train a transformer encoder model across diverse traffic patterns and intersection angles. The learned model provides speed advisories to resolve conflicts while maintaining aircraft near their desired cruising speeds. In our experiments, we evaluated encoder depths of 1, 2, and 3 layers in both structured and unstructured airspaces, and found that a single encoder configuration outperformed deeper variants, yielding near-zero near mid-air collision rates and shorter loss-of-separation infringements than the deeper configurations. Additionally, we showed that the same configuration outperforms a baseline model designed purely with attention. Together, our results suggest that the newly formulated state representation, novel design of neural network architecture, and proposed training strategy provide an adaptable and scalable decentralized solution for aircraft separation assurance in both structured and unstructured airspaces.

**arXiv ID:** 2601.04401
</details>

<details>
<summary><strong>A Novel Convex Layers Strategy for Circular Formation in Multi-Agent Systems</strong> - Gautam Kumar, Ashwini Ratnoo - [[pdf]](https://arxiv.org/pdf/2404.11351)</summary>

**Abstract:** This article considers the problem of conflict-free distribution of point-sized agents on a circular periphery encompassing all agents. The two key elements of the proposed policy include the construction of a set of convex layers (nested convex polygons) using the initial positions of the agents, and a novel search space region for each of the agents. The search space for an agent on a convex layer is defined as the region enclosed between the lines passing through the agent's position and normal to its supporting edges. Guaranteeing collision-free paths, a goal assignment policy designates a unique goal position within the search space of an agent at the initial time itself, requiring no further computation thereafter. In contrast to the existing literature, this work presents a one-shot, collision-free solution to the circular distribution problem by utilizing only the initial positions of the agents. Illustrative examples and extensive Monte-Carlo studies considering various practical attributes demonstrate the effectiveness of the proposed method.

**arXiv ID:** 2404.11351
</details>

<details>
<summary><strong>LinguaGame: A Linguistically Grounded Game-Theoretic Paradigm for Multi-Agent Dialogue Generation</strong> - Yuxiao Ye, Yiming Zhang, Yiran Ma, Huiyuan Xie, Huining Zhu, Zhiyuan Liu - [[pdf]](https://arxiv.org/pdf/2601.04516)</summary>

**Abstract:** Large Language Models (LLMs) have enabled Multi-Agent Systems (MASs) where agents interact through natural language to solve complex tasks or simulate multi-party dialogues. Recent work on LLM-based MASs has mainly focused on architecture design, such as role assignment and workflow orchestration. In contrast, this paper targets the interaction process itself, aiming to improve agents' communication efficiency by helping them convey their intended meaning more effectively through language. To this end, we propose LinguaGame, a linguistically-grounded game-theoretic paradigm for multi-agent dialogue generation. Our approach models dialogue as a signalling game over communicative intents and strategies, solved with a training-free equilibrium approximation algorithm for inference-time decision adjustment. Unlike prior game-theoretic MASs, whose game designs are often tightly coupled with task-specific objectives, our framework relies on linguistically informed reasoning with minimal task-specific coupling. Specifically, it treats dialogue as intentional and strategic communication, requiring agents to infer what others aim to achieve (intents) and how they pursue those goals (strategies). We evaluate our framework in simulated courtroom proceedings and debates, with human expert assessments showing significant gains in communication efficiency.

**arXiv ID:** 2601.04516
</details>

<details>
<summary><strong>Tool-MAD: A Multi-Agent Debate Framework for Fact Verification with Diverse Tool Augmentation and Adaptive Retrieval</strong> - Seyeon Jeong, Yeonjun Choi, JongWook Kim, Beakcheol Jang - [[pdf]](https://arxiv.org/pdf/2601.04742)</summary>

**Abstract:** Large Language Models (LLMs) suffer from hallucinations and factual inaccuracies, especially in complex reasoning and fact verification tasks. Multi-Agent Debate (MAD) systems aim to improve answer accuracy by enabling multiple LLM agents to engage in dialogue, promoting diverse reasoning and mutual verification. However, existing MAD frameworks primarily rely on internal knowledge or static documents, making them vulnerable to hallucinations. While MADKE introduces external evidence to mitigate this, its one-time retrieval mechanism limits adaptability to new arguments or emerging information during the debate. To address these limitations, We propose Tool-MAD, a multi-agent debate framework that enhances factual verification by assigning each agent a distinct external tool, such as a search API or RAG module. Tool-MAD introduces three key innovations: (1) a multi-agent debate framework where agents leverage heterogeneous external tools, encouraging diverse perspectives, (2) an adaptive query formulation mechanism that iteratively refines evidence retrieval based on the flow of the debate, and (3) the integration of Faithfulness and Answer Relevance scores into the final decision process, allowing the Judge agent to quantitatively assess the coherence and question alignment of each response and effectively detect hallucinations. Experimental results on four fact verification benchmarks demonstrate that Tool-MAD consistently outperforms state-of-the-art MAD frameworks, achieving up to 5.5% accuracy improvement. Furthermore, in medically specialized domains, Tool-MAD exhibits strong robustness and adaptability across various tool configurations and domain conditions, confirming its potential for broader real-world fact-checking applications.

**arXiv ID:** 2601.04742
</details>

<details>
<summary><strong>RAAR: Retrieval Augmented Agentic Reasoning for Cross-Domain Misinformation Detection</strong> - Zhiwei Liu, Runteng Guo, Baojie Qu, Yuechen Jiang, Min Peng, Qianqian Xie, Sophia Ananiadou - [[pdf]](https://arxiv.org/pdf/2601.04853)</summary>

**Abstract:** Cross-domain misinformation detection is challenging, as misinformation arises across domains with substantial differences in knowledge and discourse. Existing methods often rely on single-perspective cues and struggle to generalize to challenging or underrepresented domains, while reasoning large language models (LLMs), though effective on complex tasks, are limited to same-distribution data. To address these gaps, we introduce RAAR, the first retrieval-augmented agentic reasoning framework for cross-domain misinformation detection. To enable cross-domain transfer beyond same-distribution assumptions, RAAR retrieves multi-perspective source-domain evidence aligned with each target sample's semantics, sentiment, and writing style. To overcome single-perspective modeling and missing systematic reasoning, RAAR constructs verifiable multi-step reasoning paths through specialized multi-agent collaboration, where perspective-specific agents produce complementary analyses and a summary agent integrates them under verifier guidance. RAAR further applies supervised fine-tuning and reinforcement learning to train a single multi-task verifier to enhance verification and reasoning capabilities. Based on RAAR, we trained the RAAR-8b and RAAR-14b models. Evaluation on three cross-domain misinformation detection tasks shows that RAAR substantially enhances the capabilities of the base models and outperforms other cross-domain methods, advanced LLMs, and LLM-based adaptation approaches. The project will be released at this https URL.

**arXiv ID:** 2601.04853
</details>

<details>
<summary><strong>AgenticMath: Enhancing LLM Reasoning via Agentic-based Math Data Generation</strong> - Xianyang Liu, Yilin Liu, Shuai Wang, Hao Cheng, Andrew Estornell, Yuzhi Zhao, Jun Shu, Jiaheng Wei - [[pdf]](https://arxiv.org/pdf/2510.19361)</summary>

**Abstract:** The creation of high-quality datasets to improve Large Language Model (LLM) reasoning remains a significant challenge, as current methods often suffer from generating low-quality/incorrect answers and limited information richness from available data sources. To address this, we propose AgenticMath, a novel agentic method for generating high-quality mathematical question-answer pairs to enhance the supervised fine-tuning of LLMs. Our method operates through four stages: (1) Seed Question Filter that selects questions with high information richness, complexity, and clarity; (2) an Agentic Question Rephrase step that employs a multi-agent system to generate diverse, logically consistent paraphrases; (3) an Answer Augment step where rewrite answers using chain-of-thought reasoning to enhance numerical and logical correctness, without reliance on human-provided labels; and (4) a final Question and Answer Evaluation that retains only the most superior pairs. Extensive experiments demonstrate that, fine-tuning 3B-8B parameter LLMs on AgenticMath generated datasets (comprising only 30-60K math samples) achieves competitive or superior performance on diverse in domain and out-of-domain mathematical reasoning benchmarks compared to baselines trained on much more data (e.g., 400K or 2.3M samples). Our work demonstrates that targeted, high-quality data generation is a more efficient path to improving mathematical reasoning in LLMs than large-scale, low-quality alternatives.

**arXiv ID:** 2510.19361
</details>

<details>
<summary><strong>Making Tunable Parameters State-Dependent in Weather and Climate Models with Reinforcement Learning</strong> - Pritthijit Nath, Sebastian Schemm, Henry Moss, Peter Haynes, Emily Shuckburgh, Mark J. Webb - [[pdf]](https://arxiv.org/pdf/2601.04268)</summary>

**Abstract:** Weather and climate models rely on parametrisations to represent unresolved sub-grid processes. Traditional schemes rely on fixed coefficients that are weakly constrained and tuned offline, contributing to persistent biases that limit their ability to adapt to the underlying physics. This study presents a framework that learns components of parametrisation schemes online as a function of the evolving model state using reinforcement learning (RL) and evaluates the resulting RL-driven parameter updates across a hierarchy of idealised testbeds spanning a simple climate bias correction (SCBC), a radiative-convective equilibrium (RCE), and a zonal mean energy balance model (EBM) with both single-agent and federated multi-agent settings. Across nine RL algorithms, Truncated Quantile Critics (TQC), Deep Deterministic Policy Gradient (DDPG), and Twin Delayed DDPG (TD3) achieved the highest skill and the most stable convergence across configurations, with performance assessed against a static baseline using area-weighted RMSE, temperature profile and pressure-level diagnostics. For the EBM, single-agent RL outperformed static parameter tuning with the strongest gains in tropical and mid-latitude bands, while federated RL on multi-agent setups enabled geographically specialised control and faster convergence, with a six-agent DDPG configuration using frequent aggregation yielding the lowest area-weighted RMSE across the tropics and mid-latitudes. The learnt corrections were also physically meaningful as agents modulated EBM radiative parameters to reduce meridional biases, adjusted RCE lapse rates to match vertical temperature errors, and stabilised SCBC heating increments to limit drift. Overall, results highlight RL to deliver skilful state-dependent, and regime-aware parametrisations, offering a scalable pathway for online learning within numerical models.

**arXiv ID:** 2601.04268
</details>

<details>
<summary><strong>PEAR: Planner-Executor Agent Robustness Benchmark</strong> - Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing - [[pdf]](https://arxiv.org/pdf/2510.07505)</summary>

**Abstract:** Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.

**arXiv ID:** 2510.07505
</details>

</details>

<details open>
<summary><h2>Other Agent Research (11 papers)</h2></summary>

<details>
<summary><strong>Correcting Autonomous Driving Object Detection Misclassifications with Automated Commonsense Reasoning</strong> - Keegan Kimbrell, Wang Tianhao, Feng Chen, Gopal Gupta - [[pdf]](https://arxiv.org/pdf/2601.04271)</summary>

**Abstract:** Autonomous Vehicle (AV) technology has been heavily researched and sought after, yet there are no SAE Level 5 AVs available today in the marketplace. We contend that over-reliance on machine learning technology is the main reason. Use of automated commonsense reasoning technology, we believe, can help achieve SAE Level 5 autonomy. In this paper, we show how automated common- sense reasoning technology can be deployed in situations where there are not enough data samples available to train a deep learning-based AV model that can handle certain abnormal road scenarios. Specifically, we consider two situations where (i) a traffic signal is malfunctioning at an intersection and (ii) all the cars ahead are slowing down and steering away due to an unexpected obstruction (e.g., animals on the road). We show that in such situations, our commonsense reasoning-based solution accurately detects traffic light colors and obstacles not correctly captured by the AV's perception model. We also provide a pathway for efficiently invoking commonsense reasoning by measuring uncertainty in the computer vision model and using commonsense reasoning to handle uncertain sce- narios. We describe our experiments conducted using the CARLA simulator and the results obtained. The main contribution of our research is to show that automated commonsense reasoning effectively corrects AV-based object detection misclassifications and that hybrid models provide an effective pathway to improving AV perception.

**arXiv ID:** 2601.04271
</details>

<details>
<summary><strong>Higher-Order Knowledge Representations for Agentic Scientific Reasoning</strong> - Isabella A. Stewart, Markus J. Buehler - [[pdf]](https://arxiv.org/pdf/2601.04878)</summary>

**Abstract:** Scientific inquiry requires systems-level reasoning that integrates heterogeneous experimental data, cross-domain knowledge, and mechanistic evidence into coherent explanations. While Large Language Models (LLMs) offer inferential capabilities, they often depend on retrieval-augmented contexts that lack structural depth. Traditional Knowledge Graphs (KGs) attempt to bridge this gap, yet their pairwise constraints fail to capture the irreducible higher-order interactions that govern emergent physical behavior. To address this, we introduce a methodology for constructing hypergraph-based knowledge representations that faithfully encode multi-entity relationships. Applied to a corpus of ~1,100 manuscripts on biocomposite scaffolds, our framework constructs a global hypergraph of 161,172 nodes and 320,201 hyperedges, revealing a scale-free topology (power law exponent ~1.23) organized around highly connected conceptual hubs. This representation prevents the combinatorial explosion typical of pairwise expansions and explicitly preserves the co-occurrence context of scientific formulations. We further demonstrate that equipping agentic systems with hypergraph traversal tools, specifically using node-intersection constraints, enables them to bridge semantically distant concepts. By exploiting these higher-order pathways, the system successfully generates grounded mechanistic hypotheses for novel composite materials, such as linking cerium oxide to PCL scaffolds via chitosan intermediates. This work establishes a "teacherless" agentic reasoning system where hypergraph topology acts as a verifiable guardrail, accelerating scientific discovery by uncovering relationships obscured by traditional graph methods.

**arXiv ID:** 2601.04878
</details>

<details>
<summary><strong>SmartSearch: Process Reward-Guided Query Refinement for Search Agents</strong> - Tongyu Wen, Guanting Dong, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2601.04888)</summary>

**Abstract:** Large language model (LLM)-based search agents have proven promising for addressing knowledge-intensive problems by incorporating information retrieval capabilities. Existing works largely focus on optimizing the reasoning paradigms of search agents, yet the quality of intermediate search queries during reasoning remains overlooked. As a result, the generated queries often remain inaccurate, leading to unexpected retrieval results and ultimately limiting search agents' overall effectiveness. To mitigate this issue, we introduce SmartSearch, a framework built upon two key mechanisms: (1) Process rewards, which provide fine-grained supervision for the quality of each intermediate search query through Dual-Level Credit Assessment. (2) Query refinement, which promotes the optimization of query generation by selectively refining low-quality search queries and regenerating subsequent search rounds based on these refinements. To enable the search agent to progressively internalize the ability to improve query quality under the guidance of process rewards, we design a three-stage curriculum learning framework. This framework guides the agent through a progression from imitation, to alignment, and ultimately to generalization. Experimental results show that SmartSearch consistently surpasses existing baselines, and additional quantitative analyses further confirm its significant gains in both search efficiency and query quality. The code is available at this https URL.

**arXiv ID:** 2601.04888
</details>

<details>
<summary><strong>Controllable Memory Usage: Balancing Anchoring and Innovation in Long-Term Human-Agent Interaction</strong> - Muzhao Tian, Zisu Huang, Xiaohua Wang, Jingwen Xu, Zhengkang Guo, Qi Qian, Yuanzhe Shen, Kaitao Song, Jiakang Yuan, Changze Lv, Xiaoqing Zheng - [[pdf]](https://arxiv.org/pdf/2601.05107)</summary>

**Abstract:** As LLM-based agents are increasingly used in long-term interactions, cumulative memory is critical for enabling personalization and maintaining stylistic consistency. However, most existing systems adopt an ``all-or-nothing'' approach to memory usage: incorporating all relevant past information can lead to \textit{Memory Anchoring}, where the agent is trapped by past interactions, while excluding memory entirely results in under-utilization and the loss of important interaction history. We show that an agent's reliance on memory can be modeled as an explicit and user-controllable dimension. We first introduce a behavioral metric of memory dependence to quantify the influence of past interactions on current outputs. We then propose \textbf{Stee}rable \textbf{M}emory Agent, \texttt{SteeM}, a framework that allows users to dynamically regulate memory reliance, ranging from a fresh-start mode that promotes innovation to a high-fidelity mode that closely follows interaction history. Experiments across different scenarios demonstrate that our approach consistently outperforms conventional prompting and rigid memory masking strategies, yielding a more nuanced and effective control for personalized human-agent collaboration.

**arXiv ID:** 2601.05107
</details>

<details>
<summary><strong>Internal Representations as Indicators of Hallucinations in Agent Tool Selection</strong> - Kait Healy, Bharathi Srinivasan, Visakh Madathil, Jing Wu - [[pdf]](https://arxiv.org/pdf/2601.05214)</summary>

**Abstract:** Large Language Models (LLMs) have shown remarkable capabilities in tool calling and tool usage, but suffer from hallucinations where they choose incorrect tools, provide malformed parameters and exhibit 'tool bypass' behavior by performing simulations and generating outputs instead of invoking specialized tools or external systems. This undermines the reliability of LLM based agents in production systems as it leads to inconsistent results, and bypasses security and audit controls. Such hallucinations in agent tool selection require early detection and error handling. Unlike existing hallucination detection methods that require multiple forward passes or external validation, we present a computationally efficient framework that detects tool-calling hallucinations in real-time by leveraging LLMs' internal representations during the same forward pass used for generation. We evaluate this approach on reasoning tasks across multiple domains, demonstrating strong detection performance (up to 86.4\% accuracy) while maintaining real-time inference capabilities with minimal computational overhead, particularly excelling at detecting parameter-level hallucinations and inappropriate tool selections, critical for reliable agent deployment.

**arXiv ID:** 2601.05214
</details>

<details>
<summary><strong>Analyzing Message-Code Inconsistency in AI Coding Agent-Authored Pull Requests</strong> - Jingzhi Gong, Giovanni Pinna, Yixin Bian, Jie M. Zhang - [[pdf]](https://arxiv.org/pdf/2601.04886)</summary>

**Abstract:** Pull request (PR) descriptions generated by AI coding agents are the primary channel for communicating code changes to human reviewers. However, the alignment between these messages and the actual changes remains unexplored, raising concerns about the trustworthiness of AI agents. To fill this gap, we analyzed 23,247 agentic PRs across five agents using PR message-code inconsistency (PR-MCI). We contributed 974 manually annotated PRs, found 406 PRs (1.7%) exhibited high PR-MCI, and identified eight PR-MCI types, revealing that descriptions claiming unimplemented changes was the most common issue (45.4%). Statistical tests confirmed that high-MCI PRs had 51.7% lower acceptance rates (28.3% vs. 80.0%) and took 3.5x longer to merge (55.8 vs. 16.0 hours). Our findings suggest that unreliable PR descriptions undermine trust in AI agents, highlighting the need for PR-MCI verification mechanisms and improved PR generation to enable trustworthy human-AI collaboration.

**arXiv ID:** 2601.04886
</details>

<details>
<summary><strong>Character-R1: Enhancing Role-Aware Reasoning in Role-Playing Agents via RLVR</strong> - Yihong Tang, Kehai Chen, Xuefeng Bai, Benyou Wang, Zeming Liu, Haifeng Wang, Min Zhang - [[pdf]](https://arxiv.org/pdf/2601.04611)</summary>

**Abstract:** Current role-playing agents (RPAs) are typically constructed by imitating surface-level behaviors, but this approach lacks internal cognitive consistency, often causing out-of-character errors in complex situations. To address this, we propose Character-R1, a framework designed to provide comprehensive verifiable reward signals for effective role-aware reasoning, which are missing in recent studies. Specifically, our framework comprises three core designs: (1) Cognitive Focus Reward, which enforces explicit label-based analysis of 10 character elements (e.g., worldview) to structure internal cognition; (2) Reference-Guided Reward, which utilizes overlap-based metrics with reference responses as optimization anchors to enhance exploration and performance; and (3) Character-Conditioned Reward Normalization, which adjusts reward distributions based on character categories to ensure robust optimization across heterogeneous roles. Extensive experiments demonstrate that Character-R1 significantly outperforms existing methods in knowledge, memory and others.

**arXiv ID:** 2601.04611
</details>

<details>
<summary><strong>Current Agents Fail to Leverage World Model as Tool for Foresight</strong> - Cheng Qian, Emre Can Acikgoz, Bingxuan Li, Xiusi Chen, Yuji Zhang, Bingxiang He, Qinyu Luo, Dilek Hakkani-Tür, Gokhan Tur, Yunzhu Li, Heng Ji - [[pdf]](https://arxiv.org/pdf/2601.03905)</summary>

**Abstract:** Agents built on vision-language models increasingly face tasks that demand anticipating future states rather than relying on short-horizon reasoning. Generative world models offer a promising remedy: agents could use them as external simulators to foresee outcomes before acting. This paper empirically examines whether current agents can leverage such world models as tools to enhance their cognition. Across diverse agentic and visual question answering tasks, we observe that some agents rarely invoke simulation (fewer than 1%), frequently misuse predicted rollouts (approximately 15%), and often exhibit inconsistent or even degraded performance (up to 5%) when simulation is available or enforced. Attribution analysis further indicates that the primary bottleneck lies in the agents' capacity to decide when to simulate, how to interpret predicted outcomes, and how to integrate foresight into downstream reasoning. These findings underscore the need for mechanisms that foster calibrated, strategic interaction with world models, paving the way toward more reliable anticipatory cognition in future agent systems.

**arXiv ID:** 2601.03905
</details>

<details>
<summary><strong>Model of Spatial Human-Agent Interaction with Consideration for Others</strong> - Takafumi Sakamoto, Yugo Takeuchi - [[pdf]](https://arxiv.org/pdf/2601.04657)</summary>

**Abstract:** Communication robots often need to initiate conversations with people in public spaces. At the same time, such robots must not disturb pedestrians. To handle these two requirements, an agent needs to estimate the communication desires of others based on their behavior and then adjust its own communication activities accordingly. In this study, we construct a computational spatial interaction model that considers others. Consideration is expressed as a quantitative parameter: the amount of adjustment of one's internal state to the estimated internal state of the other. To validate the model, we experimented with a human and a virtual robot interacting in a VR environment. The results show that when the participant moves to the target, a virtual robot with a low consideration value inhibits the participant's movement, while a robot with a higher consideration value did not inhibit the participant's movement. When the participant approached the robot, the robot also exhibited approaching behavior, regardless of the consideration value, thus decreasing the participant's movement. These results appear to verify the proposed model's ability to clarify interactions with consideration for others.

**arXiv ID:** 2601.04657
</details>

<details>
<summary><strong>Cognitive-Hierarchy Guided End-to-End Planning for Autonomous Driving</strong> - Zhennan Wang, Jianing Teng, Canqun Xiang, Kangliang Chen, Xing Pan, Lu Deng, Weihao Gu - [[pdf]](https://arxiv.org/pdf/2505.21581)</summary>

**Abstract:** While end-to-end autonomous driving has advanced significantly, prevailing methods remain fundamentally misaligned with human cognitive principles in both perception and planning. In this paper, we propose CogAD, a novel end-to-end autonomous driving model that emulates the hierarchical cognition mechanisms of human drivers. CogAD implements dual hierarchical mechanisms: global-to-local context processing for human-like perception and intent-conditioned multi-mode trajectory generation for cognitively-inspired planning. The proposed method demonstrates three principal advantages: comprehensive environmental understanding through hierarchical perception, robust planning exploration enabled by multi-level planning, and diverse yet reasonable multi-modal trajectory generation facilitated by dual-level uncertainty modeling. Extensive experiments on nuScenes and Bench2Drive demonstrate that CogAD achieves state-of-the-art performance in end-to-end planning, exhibiting particular superiority in long-tail scenarios and robust generalization to complex real-world driving conditions.

**arXiv ID:** 2505.21581
</details>

<details>
<summary><strong>The UnScripted Trip: Fostering Policy Discussion on Future Human-Vehicle Collaboration in Autonomous Driving Through Design-Oriented Methods</strong> - Xinyan Yu, Julie Stephany Berrio Perez, Marius Hoggenmüller, Martin Tomitsch, Tram Thi Minh Tran, Stewart Worrall, Wendy Ju - [[pdf]](https://arxiv.org/pdf/2601.04601)</summary>

**Abstract:** The rapid advancement of autonomous vehicle (AV) technologies is fundamentally reshaping paradigms of human-vehicle collaboration, raising not only an urgent need for innovative design solutions but also for policies that address corresponding broader tensions in society. To bridge the gap between HCI research and policy making, this workshop will bring together researchers and practitioners in the automotive community to explore AV policy directions through collaborative speculation on the future of AVs. We designed The UnScripted Trip, a card game rooted in fictional narratives of autonomous mobility, to surface tensions around human-vehicle collaboration in future AV scenarios and to provoke critical reflections on design solutions and policy directions. Our goal is to provide an engaging, participatory space and method for automotive researchers, designers, and industry practitioners to collectively explore and shape the future of human-vehicle collaboration and its policy implications.

**arXiv ID:** 2601.04601
</details>

</details>

<details open>
<summary><h2>Reinforcement Learning (34 papers)</h2></summary>

<details>
<summary><strong>A Future Capabilities Agent for Tactical Air Traffic Control</strong> - Paul Kent, George De Ath, Martin Layton, Allen Hart, Richard Everson, Ben Carvell - [[pdf]](https://arxiv.org/pdf/2601.04285)</summary>

**Abstract:** Escalating air traffic demand is driving the adoption of automation to support air traffic controllers, but existing approaches face a trade-off between safety assurance and interpretability. Optimisation-based methods such as reinforcement learning offer strong performance but are difficult to verify and explain, while rules-based systems are transparent yet rarely check safety under uncertainty. This paper outlines Agent Mallard, a forward-planning, rules-based agent for tactical control in systemised airspace that embeds a stochastic digital twin directly into its conflict-resolution loop. Mallard operates on predefined GPS-guided routes, reducing continuous 4D vectoring to discrete choices over lanes and levels, and constructs hierarchical plans from an expert-informed library of deconfliction strategies. A depth-limited backtracking search uses causal attribution, topological plan splicing, and monotonic axis constraints to seek a complete safe plan for all aircraft, validating each candidate manoeuvre against uncertain execution scenarios (e.g., wind variation, pilot response, communication loss) before commitment.
Preliminary walkthroughs with UK controllers and initial tests in the BluebirdDT airspace digital twin indicate that Mallard's behaviour aligns with expert reasoning and resolves conflicts in simplified scenarios. The architecture is intended to combine model-based safety assessment, interpretable decision logic, and tractable computational performance in future structured en-route environments.

**arXiv ID:** 2601.04285
</details>

<details>
<summary><strong>Tape: A Cellular Automata Benchmark for Evaluating Rule-Shift Generalization in Reinforcement Learning</strong> - Enze Pan - [[pdf]](https://arxiv.org/pdf/2601.04695)</summary>

**Abstract:** We present Tape, a controlled reinforcement-learning benchmark designed to isolate out-of-distribution (OOD) failure under latent rule this http URL is derived from one-dimensional cellular automata, enabling precise train/test splits where observation and action spaces are held fixed while transition rules change. Using a reproducible evaluation pipeline, we compare model-free baselines, model-based planning with learned world models, and task-inference (meta-RL) methods. A consistent pattern emerges: methods that are strong in-distribution (ID) can collapse under heldout-rule OOD, and high-variance OOD evaluation can make rankings unstable unless experiments are sufficiently this http URL provide (i) standardized OOD protocols, (ii) statistical reporting requirements (seeds, confidence intervals, and hypothesis tests), and (iii) information-theoretic identities connecting entropy reduction to conditional mutual information and expected posterior KL divergence, clarifying what "uncertainty reduction" objectives can and cannot guarantee under rule shifts.

**arXiv ID:** 2601.04695
</details>

<details>
<summary><strong>TourPlanner: A Competitive Consensus Framework with Constraint-Gated Reinforcement Learning for Travel Planning</strong> - Yinuo Wang, Mining Tan, Wenxiang Jiao, Xiaoxi Li, Hao Wang, Xuanyu Zhang, Yuan Lu, Weiming Dong - [[pdf]](https://arxiv.org/pdf/2601.04698)</summary>

**Abstract:** Travel planning is a sophisticated decision-making process that requires synthesizing multifaceted information to construct itineraries. However, existing travel planning approaches face several challenges: (1) Pruning candidate points of interest (POIs) while maintaining a high recall rate; (2) A single reasoning path restricts the exploration capability within the feasible solution space for travel planning; (3) Simultaneously optimizing hard constraints and soft constraints remains a significant difficulty. To address these challenges, we propose TourPlanner, a comprehensive framework featuring multi-path reasoning and constraint-gated reinforcement learning. Specifically, we first introduce a Personalized Recall and Spatial Optimization (PReSO) workflow to construct spatially-aware candidate POIs' set. Subsequently, we propose Competitive consensus Chain-of-Thought (CCoT), a multi-path reasoning paradigm that improves the ability of exploring the feasible solution space. To further refine the plan, we integrate a sigmoid-based gating mechanism into the reinforcement learning stage, which dynamically prioritizes soft-constraint satisfaction only after hard constraints are met. Experimental results on travel planning benchmarks demonstrate that TourPlanner achieves state-of-the-art performance, significantly surpassing existing methods in both feasibility and user-preference alignment.

**arXiv ID:** 2601.04698
</details>

<details>
<summary><strong>ThinkDrive: Chain-of-Thought Guided Progressive Reinforcement Learning Fine-Tuning for Autonomous Driving</strong> - Chang Zhao, Zheming Yang, Yunqing Hu, Qi Guo, Zijian Wang, Pengcheng Li, Wen Ji - [[pdf]](https://arxiv.org/pdf/2601.04714)</summary>

**Abstract:** With the rapid advancement of large language models (LLMs) technologies, their application in the domain of autonomous driving has become increasingly widespread. However, existing methods suffer from unstructured reasoning, poor generalization, and misalignment with human driving intent. While Chain-of-Thought (CoT) reasoning enhances decision transparency, conventional supervised fine-tuning (SFT) fails to fully exploit its potential, and reinforcement learning (RL) approaches face instability and suboptimal reasoning depth. We propose ThinkDrive, a CoT guided progressive RL fine-tuning framework for autonomous driving that synergizes explicit reasoning with difficulty-aware adaptive policy optimization. Our method employs a two-stage training strategy. First, we perform SFT using CoT explanations. Then, we apply progressive RL with a difficulty-aware adaptive policy optimizer that dynamically adjusts learning intensity based on sample complexity. We evaluate our approach on a public dataset. The results show that ThinkDrive outperforms strong RL baselines by 1.45%, 1.95%, and 1.01% on exam, easy-exam, and accuracy, respectively. Moreover, a 2B-parameter model trained with our method surpasses the much larger GPT-4o by 3.28% on the exam metric.

**arXiv ID:** 2601.04714
</details>

<details>
<summary><strong>Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning</strong> - Yuyang Hu, Jiongnan Liu, Jiejun Tan, Yutao Zhu, Zhicheng Dou - [[pdf]](https://arxiv.org/pdf/2601.04726)</summary>

**Abstract:** Large language models (LLMs) are increasingly deployed as intelligent agents that reason, plan, and interact with their environments. To effectively scale to long-horizon scenarios, a key capability for such agents is a memory mechanism that can retain, organize, and retrieve past experiences to support downstream decision-making. However, most existing approaches organize and store memories in a flat manner and rely on simple similarity-based retrieval techniques. Even when structured memory is introduced, existing methods often struggle to explicitly capture the logical relationships among experiences or memory units. Moreover, memory access is largely detached from the constructed structure and still depends on shallow semantic retrieval, preventing agents from reasoning logically over long-horizon dependencies. In this work, we propose CompassMem, an event-centric memory framework inspired by Event Segmentation Theory. CompassMem organizes memory as an Event Graph by incrementally segmenting experiences into events and linking them through explicit logical relations. This graph serves as a logic map, enabling agents to perform structured and goal-directed navigation over memory beyond superficial retrieval, progressively gathering valuable memories to support long-horizon reasoning. Experiments on LoCoMo and NarrativeQA demonstrate that CompassMem consistently improves both retrieval and reasoning performance across multiple backbone models.

**arXiv ID:** 2601.04726
</details>

<details>
<summary><strong>APEX: Academic Poster Editing Agentic Expert</strong> - Chengxin Shi, Qinnan Cai, Zeyuan Chen, Long Zeng, Yibo Zhao, Jing Yu, Jianxiang Yu, Xiang Li - [[pdf]](https://arxiv.org/pdf/2601.04794)</summary>

**Abstract:** Designing academic posters is a labor-intensive process requiring the precise balance of high-density content and sophisticated layout. While existing paper-to-poster generation methods automate initial drafting, they are typically single-pass and non-interactive, often fail to align with complex, subjective user intent. To bridge this gap, we propose APEX (Academic Poster Editing agentic eXpert), the first agentic framework for interactive academic poster editing, supporting fine-grained control with robust multi-level API-based editing and a review-and-adjustment Mechanism. In addition, we introduce APEX-Bench, the first systematic benchmark comprising 514 academic poster editing instructions, categorized by a multi-dimensional taxonomy including operation type, difficulty, and abstraction level, constructed via reference-guided and reference-free strategies to ensure realism and diversity. We further establish a multi-dimensional VLM-as-a-judge evaluation protocol to assess instruction fulfillment, modification scope, and visual consistency & harmony. Experimental results demonstrate that APEX significantly outperforms baseline methods. Our implementation is available at this https URL.

**arXiv ID:** 2601.04794
</details>

<details>
<summary><strong>Thinking-Based Non-Thinking: Solving the Reward Hacking Problem in Training Hybrid Reasoning Models via Reinforcement Learning</strong> - Siyuan Gan, Jiaheng Liu, Boyan Wang, Tianpei Yang, Runqing Miao, Yuyao Zhang, Fanyu Meng, Junlan Feng, Linjian Meng, Jing Huo, Yang Gao - [[pdf]](https://arxiv.org/pdf/2601.04805)</summary>

**Abstract:** Large reasoning models (LRMs) have attracted much attention due to their exceptional performance. However, their performance mainly stems from thinking, a long Chain of Thought (CoT), which significantly increase computational overhead. To address this overthinking problem, existing work focuses on using reinforcement learning (RL) to train hybrid reasoning models that automatically decide whether to engage in thinking or not based on the complexity of the query. Unfortunately, using RL will suffer the the reward hacking problem, e.g., the model engages in thinking but is judged as not doing so, resulting in incorrect rewards. To mitigate this problem, existing works either employ supervised fine-tuning (SFT), which incurs high computational costs, or enforce uniform token limits on non-thinking responses, which yields limited mitigation of the problem. In this paper, we propose Thinking-Based Non-Thinking (TNT). It does not employ SFT, and sets different maximum token usage for responses not using thinking across various queries by leveraging information from the solution component of the responses using thinking. Experiments on five mathematical benchmarks demonstrate that TNT reduces token usage by around 50% compared to DeepSeek-R1-Distill-Qwen-1.5B/7B and DeepScaleR-1.5B, while significantly improving accuracy. In fact, TNT achieves the optimal trade-off between accuracy and efficiency among all tested methods. Additionally, the probability of reward hacking problem in TNT's responses, which are classified as not using thinking, remains below 10% across all tested datasets.

**arXiv ID:** 2601.04805
</details>

<details>
<summary><strong>SimuAgent: An LLM-Based Simulink Modeling Assistant Enhanced with Reinforcement Learning</strong> - Yanchang Liang, Xiaowei Zhao - [[pdf]](https://arxiv.org/pdf/2601.05187)</summary>

**Abstract:** Large language models (LLMs) have revolutionized text-based code automation, but their potential in graph-oriented engineering workflows remains under-explored. We introduce SimuAgent, an LLM-powered modeling and simulation agent tailored for Simulink. SimuAgent replaces verbose XML with a concise, dictionary-style Python representation, dramatically cutting token counts, improving interpretability, and enabling fast, in-process simulation. A lightweight plan-execute architecture, trained in two stages, equips the agent with both low-level tool skills and high-level design reasoning. To tackle sparse rewards in long-horizon tasks, we propose Reflection-GRPO (ReGRPO), which augments Group Relative Policy Optimization (GRPO) with self-reflection traces that supply rich intermediate feedback, accelerating convergence and boosting robustness. Experiments on SimuBench, our newly released benchmark comprising 5300 multi-domain modeling tasks, show that a Qwen2.5-7B model fine-tuned with SimuAgent converges faster and achieves higher modeling accuracy than standard RL baselines, and even surpasses GPT-4o when evaluated with few-shot prompting on the same benchmark. Ablations confirm that the two-stage curriculum and abstract-reconstruct data augmentation further enhance generalization. SimuAgent trains and runs entirely on-premise with modest hardware, delivering a privacy-preserving, cost-effective solution for industrial model-driven engineering. SimuAgent bridges the gap between LLMs and graphical modeling environments, offering a practical solution for AI-assisted engineering design in industrial settings.

**arXiv ID:** 2601.05187
</details>

<details>
<summary><strong>LLMs for Explainable Business Decision-Making: A Reinforcement Learning Fine-Tuning Approach</strong> - Xiang Cheng, Wen Wang, Anindya Ghose - [[pdf]](https://arxiv.org/pdf/2601.04208)</summary>

**Abstract:** Artificial Intelligence (AI) models increasingly drive high-stakes consumer interactions, yet their decision logic often remains opaque. Prevailing explainable AI techniques rely on post hoc numerical feature attributions, which fail to provide coherent narratives behind model decisions. Large language models (LLMs) present an opportunity to generate natural-language explanations, but three design challenges remain unresolved: explanations must be both decision-correct and faithful to the factors that drive the prediction; they should be able to serve multiple audiences without shifting the underlying decision rule; and they should be trained in a label-efficient way that does not depend on large corpora of human-scored explanations. To address these challenges, we introduce LEXMA (LLM-based EXplanations for Multi-Audience decisions), a reinforcement-learning-based fine-tuning framework that produces narrative-driven, audience-appropriate explanations. LEXMA combines reflection-augmented supervised fine-tuning with two stages of Group Relative Policy Optimization (GRPO). Specifically, it fine-tunes two separate parameter sets to improve decision correctness and satisfy stylistic requirements for different audiences, using reward signals that do not rely on human-annotated explanations. We instantiate LEXMA in the context of mortgage approval decisions. Results demonstrate that LEXMA yields significant improvements in predictive performance compared with other LLM baselines. Moreover, human evaluations show that expert-facing explanations generated by our approach are more risk-focused, and consumer-facing explanations are clearer, more actionable, and more polite. Our study contributes a cost-efficient, systematic LLM fine-tuning approach to enhance explanation quality for business decisions, offering strong potential for scalable deployment of transparent AI systems.

**arXiv ID:** 2601.04208
</details>

<details>
<summary><strong>Online Action-Stacking Improves Reinforcement Learning Performance for Air Traffic Control</strong> - Ben Carvell, George De Ath, Eseoghene Benjamin, Richard Everson - [[pdf]](https://arxiv.org/pdf/2601.04287)</summary>

**Abstract:** We introduce online action-stacking, an inference-time wrapper for reinforcement learning policies that produces realistic air traffic control commands while allowing training on a much smaller discrete action space. Policies are trained with simple incremental heading or level adjustments, together with an action-damping penalty that reduces instruction frequency and leads agents to issue commands in short bursts. At inference, online action-stacking compiles these bursts of primitive actions into domain-appropriate compound clearances. Using Proximal Policy Optimisation and the BluebirdDT digital twin platform, we train agents to navigate aircraft along lateral routes, manage climb and descent to target flight levels, and perform two-aircraft collision avoidance under a minimum separation constraint. In our lateral navigation experiments, action stacking greatly reduces the number of issued instructions relative to a damped baseline and achieves comparable performance to a policy trained with a 37-dimensional action space, despite operating with only five actions. These results indicate that online action-stacking helps bridge a key gap between standard reinforcement learning formulations and operational ATC requirements, and provides a simple mechanism for scaling to more complex control scenarios.

**arXiv ID:** 2601.04287
</details>

<details>
<summary><strong>Rate or Fate? RLV$^\varepsilon$R: Reinforcement Learning with Verifiable Noisy Rewards</strong> - Ali Rad, Khashayar Filom, Darioush Keivan, Peyman Mohajerin Esfahani, Ehsan Kamalinejad - [[pdf]](https://arxiv.org/pdf/2601.04411)</summary>

**Abstract:** Reinforcement learning with verifiable rewards (RLVR) is a simple but powerful paradigm for training LLMs: sample a completion, verify it, and update. In practice, however, the verifier is almost never clean--unit tests probe only limited corner cases; human and synthetic labels are imperfect; and LLM judges (e.g., RLAIF) are noisy and can be exploited--and this problem worsens on harder domains (especially coding) where tests are sparse and increasingly model-generated. We ask a pragmatic question: does the verification noise merely slow down the learning (rate), or can it flip the outcome (fate)?
To address this, we develop an analytically tractable multi-armed bandit view of RLVR dynamics, instantiated with GRPO and validated in controlled experiments. Modeling false positives and false negatives and grouping completions into recurring reasoning modes yields a replicator-style (natural-selection) flow on the probability simplex. The dynamics decouples into within-correct-mode competition and a one-dimensional evolution for the mass on incorrect modes, whose drift is determined solely by Youden's index J=TPR-FPR. This yields a sharp phase transition: when J>0, the incorrect mass is driven toward extinction (learning); when J=0, the process is neutral; and when J<0, incorrect modes amplify until they dominate (anti-learning and collapse). In the learning regime J>0, noise primarily rescales convergence time ("rate, not fate"). Experiments on verifiable programming tasks under synthetic noise reproduce the predicted J=0 boundary. Beyond noise, the framework offers a general lens for analyzing RLVR stability, convergence, and algorithmic interventions.

**arXiv ID:** 2601.04411
</details>

<details>
<summary><strong>Vision-Language Agents for Interactive Forest Change Analysis</strong> - James Brock, Ce Zhang, Nantheera Anantrasirichai - [[pdf]](https://arxiv.org/pdf/2601.04497)</summary>

**Abstract:** Modern forest monitoring workflows increasingly benefit from the growing availability of high-resolution satellite imagery and advances in deep learning. Two persistent challenges in this context are accurate pixel-level change detection and meaningful semantic change captioning for complex forest dynamics. While large language models (LLMs) are being adapted for interactive data exploration, their integration with vision-language models (VLMs) for remote sensing image change interpretation (RSICI) remains underexplored. To address this gap, we introduce an LLM-driven agent for integrated forest change analysis that supports natural language querying across multiple RSICI tasks. The proposed system builds upon a multi-level change interpretation (MCI) vision-language backbone with LLM-based orchestration. To facilitate adaptation and evaluation in forest environments, we further introduce the Forest-Change dataset, which comprises bi-temporal satellite imagery, pixel-level change masks, and multi-granularity semantic change captions generated using a combination of human annotation and rule-based methods. Experimental results show that the proposed system achieves mIoU and BLEU-4 scores of 67.10% and 40.17% on the Forest-Change dataset, and 88.13% and 34.41% on LEVIR-MCI-Trees, a tree-focused subset of LEVIR-MCI benchmark for joint change detection and captioning. These results highlight the potential of interactive, LLM-driven RSICI systems to improve accessibility, interpretability, and efficiency of forest change analysis. All data and code are publicly available at this https URL.

**arXiv ID:** 2601.04497
</details>

<details>
<summary><strong>TSSR: Two-Stage Swap-Reward-Driven Reinforcement Learning for Character-Level SMILES Generation</strong> - Jacob Ede Levine, Yun Lyan Luo, Sai Chandra Kosaraju - [[pdf]](https://arxiv.org/pdf/2601.04521)</summary>

**Abstract:** The design of reliable, valid, and diverse molecules is fundamental to modern drug discovery, as improved molecular generation supports efficient exploration of the chemical space for potential drug candidates and reduces the cost of early design efforts. Despite these needs, current chemical language models that generate molecules as SMILES strings are vulnerable to compounding token errors: many samples are unparseable or chemically implausible, and hard constraints meant to prevent failure can restrict exploration. To address this gap, we introduce TSSR, a Two-Stage, Swap-Reward-driven reinforcement learning (RL) framework for character-level SMILES generation. Stage one rewards local token swaps that repair syntax, promoting transitions from invalid to parseable strings. Stage two provides chemistry-aware feedback from RDKit diagnostics, rewarding reductions in valence, aromaticity, and connectivity issues. The reward decomposes into interpretable terms (swap efficiency, error reduction, distance to validity), is model agnostic, and requires no task-specific labels or hand-crafted grammars. We evaluated TSSR on the MOSES benchmark using a GRU policy trained with PPO in both pure RL (P-RL) from random initialization and fine-tuning RL (F-RL) starting from a pretrained chemical language model, assessing 10,000 generated SMILES per run. In P-RL, TSSR significantly improves syntactic validity, chemical validity, and novelty. In F-RL, TSSR preserves drug-likeness and synthesizability while increasing validity and novelty. Token-level analysis shows that syntax edits and chemistry fixes act jointly to reduce RDKit detected errors. TSSR converts a sparse terminal objective into a denser and more interpretable reward, improving both syntactic and chemical quality without reducing diversity. TSSR is dataset-agnostic and can be adapted to various reinforcement learning approaches.

**arXiv ID:** 2601.04521
</details>

<details>
<summary><strong>Optimizing Path Planning using Deep Reinforcement Learning for UGVs in Precision Agriculture</strong> - Laukik Patade, Rohan Rane, Sandeep Pillai - [[pdf]](https://arxiv.org/pdf/2601.04668)</summary>

**Abstract:** This study focuses on optimizing path planning for unmanned ground vehicles (UGVs) in precision agriculture using deep reinforcement learning (DRL) techniques in continuous action spaces. The research begins with a review of traditional grid-based methods, such as A* and Dijkstra's algorithms, and discusses their limitations in dynamic agricultural environments, highlighting the need for adaptive learning strategies. The study then explores DRL approaches, including Deep Q-Networks (DQN), which demonstrate improved adaptability and performance in two-dimensional simulations. Enhancements such as Double Q-Networks and Dueling Networks are evaluated to further improve decision-making. Building on these results, the focus shifts to continuous action space models, specifically Deep Deterministic Policy Gradient (DDPG) and Twin Delayed Deep Deterministic Policy Gradient (TD3), which are tested in increasingly complex environments. Experiments conducted in a three-dimensional environment using ROS and Gazebo demonstrate the effectiveness of continuous DRL algorithms in navigating dynamic agricultural scenarios. Notably, the pretrained TD3 agent achieves a 95 percent success rate in dynamic environments, demonstrating the robustness of the proposed approach in handling moving obstacles while ensuring safety for both crops and the robot.

**arXiv ID:** 2601.04668
</details>

<details>
<summary><strong>AgentOCR: Reimagining Agent History via Optical Self-Compression</strong> - Lang Feng, Fuchao Yang, Feng Chen, Xin Cheng, Haiyang Xu, Zhenglin Wan, Ming Yan, Bo An - [[pdf]](https://arxiv.org/pdf/2601.04786)</summary>

**Abstract:** Recent advances in large language models (LLMs) enable agentic systems trained with reinforcement learning (RL) over multi-turn interaction trajectories, but practical deployment is bottlenecked by rapidly growing textual histories that inflate token budgets and memory usage. We introduce AgentOCR, a framework that exploits the superior information density of visual tokens by representing the accumulated observation-action history as a compact rendered image. To make multi-turn rollouts scalable, AgentOCR proposes segment optical caching. By decomposing history into hashable segments and maintaining a visual cache, this mechanism eliminates redundant re-rendering. Beyond fixed rendering, AgentOCR introduces agentic self-compression, where the agent actively emits a compression rate and is trained with compression-aware reward to adaptively balance task success and token efficiency. We conduct extensive experiments on challenging agentic benchmarks, ALFWorld and search-based QA. Remarkably, results demonstrate that AgentOCR preserves over 95\% of text-based agent performance while substantially reducing token consumption (>50\%), yielding consistent token and memory efficiency. Our further analysis validates a 20x rendering speedup from segment optical caching and the effective strategic balancing of self-compression.

**arXiv ID:** 2601.04786
</details>

<details>
<summary><strong>On the Hidden Objective Biases of Group-based Reinforcement Learning</strong> - Aleksandar Fontana, Marco Simoni, Giulio Rossolini, Andrea Saracino, Paolo Mori - [[pdf]](https://arxiv.org/pdf/2601.05002)</summary>

**Abstract:** Group-based reinforcement learning methods, like Group Relative Policy Optimization (GRPO), are widely used nowadays to post-train large language models. Despite their empirical success, they exhibit structural mismatches between reward optimization and the underlying training objective. In this paper, we present a theoretical analysis of GRPO style methods by studying them within a unified surrogate formulation. This perspective reveals recurring properties that affect all the methods under analysis: (i) non-uniform group weighting induces systematic gradient biases on shared prefix tokens; (ii) interactions with the AdamW optimizer make training dynamics largely insensitive to reward scaling; and (iii) optimizer momentum can push policy updates beyond the intended clipping region under repeated optimization steps. We believe that these findings highlight fundamental limitations of current approaches and provide principled guidance for the design of future formulations.

**arXiv ID:** 2601.05002
</details>

<details>
<summary><strong>Safe Continual Reinforcement Learning Methods for Nonstationary Environments. Towards a Survey of the State of the Art</strong> - Timofey Tomashevskiy - [[pdf]](https://arxiv.org/pdf/2601.05152)</summary>

**Abstract:** This work provides a state-of-the-art survey of continual safe online reinforcement learning (COSRL) methods. We discuss theoretical aspects, challenges, and open questions in building continual online safe reinforcement learning algorithms. We provide the taxonomy and the details of continual online safe reinforcement learning methods based on the type of safe learning mechanism that takes adaptation to nonstationarity into account. We categorize safety constraints formulation for online reinforcement learning algorithms, and finally, we discuss prospects for creating reliable, safe online learning algorithms.
Keywords: safe RL in nonstationary environments, safe continual reinforcement learning under nonstationarity, HM-MDP, NSMDP, POMDP, safe POMDP, constraints for continual learning, safe continual reinforcement learning review, safe continual reinforcement learning survey, safe continual reinforcement learning, safe online learning under distribution shift, safe continual online adaptation, safe reinforcement learning, safe exploration, safe adaptation, constrained Markov decision processes, safe reinforcement learning, partially observable Markov decision process, safe reinforcement learning and hidden Markov decision processes, Safe Online Reinforcement Learning, safe online reinforcement learning, safe online reinforcement learning, safe meta-learning, safe meta-reinforcement learning, safe context-based reinforcement learning, formulating safety constraints for continual learning

**arXiv ID:** 2601.05152
</details>

<details>
<summary><strong>Nalar: An agent serving framework</strong> - Marco Laju, Donghyun Son, Saurabh Agarwal, Nitin Kedia, Myungjin Lee, Jayanth Srinivasa, Aditya Akella - [[pdf]](https://arxiv.org/pdf/2601.05109)</summary>

**Abstract:** LLM-driven agentic applications increasingly automate complex, multi-step tasks, but serving them efficiently remains challenging due to heterogeneous components, dynamic and model-driven control flow, long-running state, and unpredictable latencies. Nalar is a ground-up agent-serving framework that cleanly separates workflow specification from execution while providing the runtime visibility and control needed for robust performance. Nalar preserves full Python expressiveness, using lightweight auto-generated stubs that turn agent and tool invocations into futures carrying dependency and context metadata. A managed state layer decouples logical state from physical placement, enabling safe reuse, migration, and consistent retry behavior. A two-level control architecture combines global policy computation with local event-driven enforcement to support adaptive routing, scheduling, and resource management across evolving workflows. Together, these mechanisms allow Nalar to deliver scalable, efficient, and policy-driven serving of heterogeneous agentic applications without burdening developers with orchestration logic. Across three agentic workloads, Nalar cuts tail latency by 34--74\%, achieves up to $2.9\times$ speedups, sustains 80 RPS where baselines fail, and scales to 130K futures with sub-500 ms control overhead.

**arXiv ID:** 2601.05109
</details>

<details>
<summary><strong>GRACE: Reinforcement Learning for Grounded Response and Abstention under Contextual Evidence</strong> - Yibo Zhao, Jiapeng Zhu, Zichen Ding, Xiang Li - [[pdf]](https://arxiv.org/pdf/2601.04525)</summary>

**Abstract:** Retrieval-Augmented Generation (RAG) integrates external knowledge to enhance Large Language Models (LLMs), yet systems remain susceptible to two critical flaws: providing correct answers without explicit grounded evidence and producing fabricated responses when the retrieved context is insufficient. While prior research has addressed these issues independently, a unified framework that integrates evidence-based grounding and reliable abstention is currently lacking. In this paper, we propose GRACE, a reinforcement-learning framework that simultaneously mitigates both types of flaws. GRACE employs a data construction method that utilizes heterogeneous retrievers to generate diverse training samples without manual annotation. A multi-stage gated reward function is then employed to train the model to assess evidence sufficiency, extract key supporting evidence, and provide answers or explicitly abstain. Experimental results on two benchmarks demonstrate that GRACE achieves state-of-the-art overall accuracy and strikes a favorable balance between accurate response and rejection, while requiring only 10% of the annotation costs of prior methods. Our code is available at this https URL..

**arXiv ID:** 2601.04525
</details>

<details>
<summary><strong>Aligning Text, Code, and Vision: A Multi-Objective Reinforcement Learning Framework for Text-to-Visualization</strong> - Mizanur Rahman, Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Shafiq Joty, Enamul Hoque - [[pdf]](https://arxiv.org/pdf/2601.04582)</summary>

**Abstract:** Text-to-Visualization (Text2Vis) systems translate natural language queries over tabular data into concise answers and executable visualizations. While closed-source LLMs generate functional code, the resulting charts often lack semantic alignment and clarity, qualities that can only be assessed post-execution. Open-source models struggle even more, frequently producing non-executable or visually poor outputs. Although supervised fine-tuning can improve code executability, it fails to enhance overall visualization quality, as traditional SFT loss cannot capture post-execution feedback. To address this gap, we propose RL-Text2Vis, the first reinforcement learning framework for Text2Vis generation. Built on Group Relative Policy Optimization (GRPO), our method uses a novel multi-objective reward that jointly optimizes textual accuracy, code validity, and visualization quality using post-execution feedback. By training Qwen2.5 models (7B and 14B), RL-Text2Vis achieves a 22% relative improvement in chart quality over GPT-4o on the Text2Vis benchmark and boosts code execution success from 78% to 97% relative to its zero-shot baseline. Our models significantly outperform strong zero-shot and supervised baselines and also demonstrate robust generalization to out-of-domain datasets like VIS-Eval and NVBench. These results establish GRPO as an effective strategy for structured, multimodal reasoning in visualization generation. We release our code at this https URL.

**arXiv ID:** 2601.04582
</details>

<details>
<summary><strong>Mind2Report: A Cognitive Deep Research Agent for Expert-Level Commercial Report Synthesis</strong> - Mingyue Cheng, Daoyu Wang, Qi Liu, Shuo Yu, Xiaoyu Tao, Yuqian Wang, Chengzhong Chu, Yu Duan, Mingkang Long, Enhong Chen - [[pdf]](https://arxiv.org/pdf/2601.04879)</summary>

**Abstract:** Synthesizing informative commercial reports from massive and noisy web sources is critical for high-stakes business decisions. Although current deep research agents achieve notable progress, their reports still remain limited in terms of quality, reliability, and coverage. In this work, we propose Mind2Report, a cognitive deep research agent that emulates the commercial analyst to synthesize expert-level reports. Specifically, it first probes fine-grained intent, then searches web sources and records distilled information on the fly, and subsequently iteratively synthesizes the report. We design Mind2Report as a training-free agentic workflow that augments general large language models (LLMs) with dynamic memory to support these long-form cognitive processes. To rigorously evaluate Mind2Report, we further construct QRC-Eval comprising 200 real-world commercial tasks and establish a holistic evaluation strategy to assess report quality, reliability, and coverage. Experiments demonstrate that Mind2Report outperforms leading baselines, including OpenAI and Gemini deep research agents. Although this is a preliminary study, we expect it to serve as a foundation for advancing the future design of commercial deep research agents. Our code and data are available at this https URL.

**arXiv ID:** 2601.04879
</details>

<details>
<summary><strong>Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems</strong> - Jihao Zhao, Ding Chen, Zhaoxin Fan, Kerun Xu, Mengting Hu, Bo Tang, Feiyu Xiong, Zhiyu li - [[pdf]](https://arxiv.org/pdf/2601.05171)</summary>

**Abstract:** Existing long-term personalized dialogue systems struggle to reconcile unbounded interaction streams with finite context constraints, often succumbing to memory noise accumulation, reasoning degradation, and persona inconsistency. To address these challenges, this paper proposes Inside Out, a framework that utilizes a globally maintained PersonaTree as the carrier of long-term user profiling. By constraining the trunk with an initial schema and updating the branches and leaves, PersonaTree enables controllable growth, achieving memory compression while preserving consistency. Moreover, we train a lightweight MemListener via reinforcement learning with process-based rewards to produce structured, executable, and interpretable {ADD, UPDATE, DELETE, NO_OP} operations, thereby supporting the dynamic evolution of the personalized tree. During response generation, PersonaTree is directly leveraged to enhance outputs in latency-sensitive scenarios; when users require more details, the agentic mode is triggered to introduce details on-demand under the constraints of the PersonaTree. Experiments show that PersonaTree outperforms full-text concatenation and various personalized memory systems in suppressing contextual noise and maintaining persona consistency. Notably, the small MemListener model achieves memory-operation decision performance comparable to, or even surpassing, powerful reasoning models such as DeepSeek-R1-0528 and Gemini-3-Pro.

**arXiv ID:** 2601.05171
</details>

<details>
<summary><strong>Agri-R1: Empowering Generalizable Agricultural Reasoning in Vision-Language Models with Reinforcement Learning</strong> - Wentao Zhang, Lifei Wang, Lina Lu, MingKun Xu, Shangyang Li, Yanchao Yang, Tao Fang - [[pdf]](https://arxiv.org/pdf/2601.04672)</summary>

**Abstract:** Agricultural disease diagnosis challenges VLMs, as conventional fine-tuning requires extensive labels, lacks interpretability, and generalizes poorly. While reasoning improves model robustness, existing methods rely on costly expert annotations and rarely address the open-ended, diverse nature of agricultural queries. To address these limitations, we propose \textbf{Agri-R1}, a reasoning-enhanced large model for agriculture. Our framework automates high-quality reasoning data generation via vision-language synthesis and LLM-based filtering, using only 19\% of available samples. Training employs Group Relative Policy Optimization (GRPO) with a novel proposed reward function that integrates domain-specific lexicons and fuzzy matching to assess both correctness and linguistic flexibility in open-ended responses. Evaluated on CDDMBench, our resulting 3B-parameter model achieves performance competitive with 7B- to 13B-parameter baselines, showing a +23.2\% relative gain in disease recognition accuracy, +33.3\% in agricultural knowledge QA, and a +26.10-point improvement in cross-domain generalization over standard fine-tuning. Ablation studies confirm that the synergy between structured reasoning data and GRPO-driven exploration underpins these gains, with benefits scaling as question complexity increases.

**arXiv ID:** 2601.04672
</details>

<details>
<summary><strong>Think Natively: Unlocking Multilingual Reasoning with Consistency-Enhanced Reinforcement Learning</strong> - Xue Zhang, Yunlong Liang, Fandong Meng, Songming Zhang, Kaiyu Huang, Yufeng Chen, Jinan Xu, Jie Zhou - [[pdf]](https://arxiv.org/pdf/2510.07300)</summary>

**Abstract:** Large Reasoning Models (LRMs) have achieved remarkable performance on complex reasoning tasks by adopting the ``think-then-answer'' paradigm, which enhances both accuracy and interpretability. However, current LRMs exhibit two critical limitations when processing non-English languages: (1) They often struggle to maintain input-output language consistency; (2) They generally perform poorly with wrong reasoning paths and lower answer accuracy compared to English. These limitations significantly compromise the interpretability of reasoning processes and degrade the user experience for non-English speakers, hindering the global deployment of LRMs. To address these limitations, we propose M-Thinker, which is trained by the GRPO algorithm that involves a Language Consistency (LC) reward and a novel Cross-lingual Thinking Alignment (CTA) reward. Specifically, the LC reward defines a strict constraint on the language consistency between the input, thought, and answer. Besides, the CTA reward compares the model's non-English reasoning paths with its English reasoning path to transfer its own reasoning capability from English to non-English languages. Through an iterative RL procedure, our M-Thinker-1.5B/4B/7B models not only achieve nearly 100% language consistency and superior performance on two multilingual benchmarks (MMATH and PolyMath), but also exhibit excellent generalization on out-of-domain languages.

**arXiv ID:** 2510.07300
</details>

<details>
<summary><strong>InfiniteWeb: Scalable Web Environment Synthesis for GUI Agent Training</strong> - Ziyun Zhang, Zezhou Wang, Xiaoyi Zhang, Zongyu Guo, Jiahao Li, Bin Li, Yan Lu - [[pdf]](https://arxiv.org/pdf/2601.04126)</summary>

**Abstract:** GUI agents that interact with graphical interfaces on behalf of users represent a promising direction for practical AI assistants. However, training such agents is hindered by the scarcity of suitable environments. We present InfiniteWeb, a system that automatically generates functional web environments at scale for GUI agent training. While LLMs perform well on generating a single webpage, building a realistic and functional website with many interconnected pages faces challenges. We address these challenges through unified specification, task-centric test-driven development, and a combination of website seed with reference design image to ensure diversity. Our system also generates verifiable task evaluators enabling dense reward signals for reinforcement learning. Experiments show that InfiniteWeb surpasses commercial coding agents at realistic website construction, and GUI agents trained on our generated environments achieve significant performance improvements on OSWorld and Online-Mind2Web, demonstrating the effectiveness of proposed system.

**arXiv ID:** 2601.04126
</details>

<details>
<summary><strong>Survival Dynamics of Neural and Programmatic Policies in Evolutionary Reinforcement Learning</strong> - Anton Roupassov-Ruiz, Yiyang Zuo - [[pdf]](https://arxiv.org/pdf/2601.04365)</summary>

**Abstract:** In evolutionary reinforcement learning tasks (ERL), agent policies are often encoded as small artificial neural networks (NERL). Such representations lack explicit modular structure, limiting behavioral interpretation. We investigate whether programmatic policies (PERL), implemented as soft, differentiable decision lists (SDDL), can match the performance of NERL. To support reproducible evaluation, we provide the first fully specified and open-source reimplementation of the classic 1992 Artificial Life (ALife) ERL testbed. We conduct a rigorous survival analysis across 4000 independent trials utilizing Kaplan-Meier curves and Restricted Mean Survival Time (RMST) metrics absent in the original study. We find a statistically significant difference in survival probability between PERL and NERL. PERL agents survive on average 201.69 steps longer than NERL agents. Moreover, SDDL agents using learning alone (no evolution) survive on average 73.67 steps longer than neural agents using both learning and evaluation. These results demonstrate that programmatic policies can exceed the survival performance of neural policies in ALife.

**arXiv ID:** 2601.04365
</details>

<details>
<summary><strong>Multiagent Reinforcement Learning with Neighbor Action Estimation</strong> - Zhenglong Luo, Zhiyong Chen, Aoxiang Liu - [[pdf]](https://arxiv.org/pdf/2601.04511)</summary>

**Abstract:** Multiagent reinforcement learning, as a prominent intelligent paradigm, enables collaborative decision-making within complex systems. However, existing approaches often rely on explicit action exchange between agents to evaluate action value functions, which is frequently impractical in real-world engineering environments due to communication constraints, latency, energy consumption, and reliability requirements. From an artificial intelligence perspective, this paper proposes an enhanced multiagent reinforcement learning framework that employs action estimation neural networks to infer agent behaviors. By integrating a lightweight action estimation module, each agent infers neighboring agents' behaviors using only locally observable information, enabling collaborative policy learning without explicit action sharing. This approach is fully compatible with standard TD3 algorithms and scalable to larger multiagent systems. At the engineering application level, this framework has been implemented and validated in dual-arm robotic manipulation tasks: two robotic arms collaboratively lift objects. Experimental results demonstrate that this approach significantly enhances the robustness and deployment feasibility of real-world robotic systems while reducing dependence on information infrastructure. Overall, this research advances the development of decentralized multiagent artificial intelligence systems while enabling AI to operate effectively in dynamic, information-constrained real-world environments.

**arXiv ID:** 2601.04511
</details>

<details>
<summary><strong>BraVE: Offline Reinforcement Learning for Discrete Combinatorial Action Spaces</strong> - Matthew Landers, Taylor W. Killian, Hugo Barnes, Thomas Hartvigsen, Afsaneh Doryab - [[pdf]](https://arxiv.org/pdf/2410.21151)</summary>

**Abstract:** Offline reinforcement learning in high-dimensional, discrete action spaces is challenging due to the exponential scaling of the joint action space with the number of sub-actions and the complexity of modeling sub-action dependencies. Existing methods either exhaustively evaluate the action space, making them computationally infeasible, or factorize Q-values, failing to represent joint sub-action effects. We propose Branch Value Estimation (BraVE), a value-based method that uses tree-structured action traversal to evaluate a linear number of joint actions while preserving dependency structure. BraVE outperforms prior offline RL methods by up to $20\times$ in environments with over four million actions.

**arXiv ID:** 2410.21151
</details>

<details>
<summary><strong>ReLA: Representation Learning and Aggregation for Job Scheduling with Reinforcement Learning</strong> - Zhengyi Kwan, Wei Zhang, Aik Beng Ng, Zhengkui Wang, Simon See - [[pdf]](https://arxiv.org/pdf/2601.03646)</summary>

**Abstract:** Job scheduling is widely used in real-world manufacturing systems to assign ordered job operations to machines under various constraints. Existing solutions remain limited by long running time or insufficient schedule quality, especially when problem scale increases. In this paper, we propose ReLA, a reinforcement-learning (RL) scheduler built on structured representation learning and aggregation. ReLA first learns diverse representations from scheduling entities, including job operations and machines, using two intra-entity learning modules with self-attention and convolution and one inter-entity learning module with cross-attention. These modules are applied in a multi-scale architecture, and their outputs are aggregated to support RL decision-making. Across experiments on small, medium, and large job instances, ReLA achieves the best makespan in most tested settings over the latest solutions. On non-large instances, ReLA reduces the optimality gap of the SOTA baseline by 13.0%, while on large-scale instances it reduces the gap by 78.6%, with the average optimality gaps lowered to 7.3% and 2.1%, respectively. These results confirm that ReLA's learned representations and aggregation provide strong decision support for RL scheduling, and enable fast job completion and decision-making for real-world applications.

**arXiv ID:** 2601.03646
</details>

<details>
<summary><strong>Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots</strong> - Chenhao Li, Andreas Krause, Marco Hutter - [[pdf]](https://arxiv.org/pdf/2504.16680)</summary>

**Abstract:** Reinforcement Learning (RL) has achieved impressive results in robotics, yet high-performing pipelines remain highly task-specific, with little reuse of prior data. Offline Model-based RL (MBRL) offers greater data efficiency by training policies entirely from existing datasets, but suffers from compounding errors and distribution shift in long-horizon rollouts. Although existing methods have shown success in controlled simulation benchmarks, robustly applying them to the noisy, biased, and partially observed datasets typical of real-world robotics remains challenging. We present a principled pipeline for making offline MBRL effective on physical robots. Our RWM-U extends autoregressive world models with epistemic uncertainty estimation, enabling temporally consistent multi-step rollouts with uncertainty effectively propagated over long horizons. We combine RWM-U with MOPO-PPO, which adapts uncertainty-penalized policy optimization to the stable, on-policy PPO framework for real-world control. We evaluate our approach on diverse manipulation and locomotion tasks in simulation and on real quadruped and humanoid, training policies entirely from offline datasets. The resulting policies consistently outperform model-free and uncertainty-unaware model-based baselines, and fusing real-world data in model learning further yields robust policies that surpass online model-free baselines trained solely in simulation.

**arXiv ID:** 2504.16680
</details>

<details>
<summary><strong>Solving Robotics Tasks with Prior Demonstration via Exploration-Efficient Deep Reinforcement Learning</strong> - Chengyandan Shen, Christoffer Sloth - [[pdf]](https://arxiv.org/pdf/2509.04069)</summary>

**Abstract:** This paper proposes an exploration-efficient Deep Reinforcement Learning with Reference policy (DRLR) framework for learning robotics tasks that incorporates demonstrations. The DRLR framework is developed based on an algorithm called Imitation Bootstrapped Reinforcement Learning (IBRL). We propose to improve IBRL by modifying the action selection module. The proposed action selection module provides a calibrated Q-value, which mitigates the bootstrapping error that otherwise leads to inefficient exploration. Furthermore, to prevent the RL policy from converging to a sub-optimal policy, SAC is used as the RL policy instead of TD3. The effectiveness of our method in mitigating bootstrapping error and preventing overfitting is empirically validated by learning two robotics tasks: bucket loading and open drawer, which require extensive interactions with the environment. Simulation results also demonstrate the robustness of the DRLR framework across tasks with both low and high state-action dimensions, and varying demonstration qualities. To evaluate the developed framework on a real-world industrial robotics task, the bucket loading task is deployed on a real wheel loader. The sim2real results validate the successful deployment of the DRLR framework.

**arXiv ID:** 2509.04069
</details>

<details>
<summary><strong>Cells on Autopilot: Adaptive Cell (Re)Selection via Reinforcement Learning</strong> - Marvin Illian, Ramin Khalili, Antonio A. de A. Rocha, Lin Wang - [[pdf]](https://arxiv.org/pdf/2601.04083)</summary>

**Abstract:** The widespread deployment of 5G networks, together with the coexistence of 4G/LTE networks, provides mobile devices a diverse set of candidate cells to connect to. However, associating mobile devices to cells to maximize overall network performance, a.k.a. cell (re)selection, remains a key challenge for mobile operators. Today, cell (re)selection parameters are typically configured manually based on operator experience and rarely adapted to dynamic network conditions. In this work, we ask: Can an agent automatically learn and adapt cell (re)selection parameters to consistently improve network performance? We present a reinforcement learning (RL)-based framework called CellPilot that adaptively tunes cell (re)selection parameters by learning spatiotemporal patterns of mobile network dynamics. Our study with real-world data demonstrates that even a lightweight RL agent can outperform conventional heuristic reconfigurations by up to 167%, while generalizing effectively across different network scenarios. These results indicate that data-driven approaches can significantly improve cell (re)selection configurations and enhance mobile network performance.

**arXiv ID:** 2601.04083
</details>

<details>
<summary><strong>Autonomous Reasoning for Spacecraft Control: A Large Language Model Framework with Group Relative Policy Optimization</strong> - Amit Jain, Richard Linares - [[pdf]](https://arxiv.org/pdf/2601.04334)</summary>

**Abstract:** This paper presents a learning-based guidance-and-control approach that couples a reasoning-enabled Large Language Model (LLM) with Group Relative Policy Optimization (GRPO). A two-stage procedure consisting of Supervised Fine-Tuning (SFT) to learn formatting and control primitives, followed by GRPO for interaction-driven policy improvement, trains controllers for each environment. The framework is demonstrated on four control problems spanning a gradient of dynamical complexity, from canonical linear systems through nonlinear oscillatory dynamics to three-dimensional spacecraft attitude control with gyroscopic coupling and thrust constraints. Results demonstrate that an LLM with explicit reasoning, optimized via GRPO, can synthesize feasible stabilizing policies under consistent training settings across both linear and nonlinear systems. The two-stage training methodology enables models to generate control sequences while providing human-readable explanations of their decision-making process. This work establishes a foundation for applying GRPO-based reasoning to autonomous control systems, with potential applications in aerospace and other safety-critical domains.

**arXiv ID:** 2601.04334
</details>

<details>
<summary><strong>DAVOS: An Autonomous Vehicle Operating System in the Vehicle Computing Era</strong> - Yuxin Wang, Yuankai He, Boyang Tian, Lichen Xian, Weisong Shi - [[pdf]](https://arxiv.org/pdf/2601.05072)</summary>

**Abstract:** Vehicle computing represents a fundamental shift in how autonomous vehicles are designed and deployed, transforming them from isolated transportation systems into mobile computing platforms that support both safety-critical, real-time driving and data-centric services. In this setting, vehicles simultaneously support real-time driving pipelines and a growing set of data-driven applications, placing increased responsibility on the vehicle operating system to coordinate computation, data movement, storage, and access. These demands highlight recurring system considerations related to predictable execution, data and execution protection, efficient handling of high-rate sensor data, and long-term system evolvability, commonly summarized as Safety, Security, Efficiency, and Extensibility (SSEE). Existing vehicle operating systems and runtimes address these concerns in isolation, resulting in fragmented software stacks that limit coordination between autonomy workloads and vehicle data services. This paper presents DAVOS, the Delaware Autonomous Vehicle Operating System, a unified vehicle operating system architecture designed for the vehicle computing context. DAVOS provides a cohesive operating system foundation that supports both real-time autonomy and extensible vehicle computing within a single system framework.

**arXiv ID:** 2601.05072
</details>

</details>

---

*This list is automatically generated daily using arXiv web scraping*
