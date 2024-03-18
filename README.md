# Awesome-LLMs-ICLR'24
 It is a comprehensive resource hub compiling all LLM papers accepted at the International Conference on Learning Representations (ICLR) in 2024.

### Breaking Physical and Linguistic Borders: Multilingual Federated Prompt Tuning for Low-Resource Languages
> By enabling privacy-preserving fine-tuning, the Federated Prompt Tuning Paradigm overcomes obstacles in deploying multilingual LLMs for low-resource languages, fostering data efficiency, mutual language enhancement, and broader accessibility.

<details>
<summary>Details</summary>
- **Abstract**: Pretrained large language models (LLMs) have emerged as a cornerstone in modern natural language processing, with their utility expanding to various applications and languages. However, the fine-tuning of multilingual LLMs, particularly for low-resource languages, is fraught with challenges steming from data-sharing restrictions (the physical border) and from the inherent linguistic differences (the linguistic border). These barriers hinder users of various languages, especially those in low-resource regions, from fully benefiting from the advantages of LLMs.  To overcome these challenges, we propose the Federated Prompt Tuning Paradigm for Multilingual Scenarios, which leverages parameter-efficient fine-tuning in a manner that preserves user privacy. We have designed a comprehensive set of experiments and introduced the concept of "language distance" to highlight the several strengths of this paradigm. Even under computational constraints, our method not only bolsters data efficiency but also facilitates mutual enhancements across languages, particularly benefiting low-resource ones. Compared to traditional local crosslingual transfer tuning methods, our approach achieves a 6.9% higher accuracy, reduces the training parameters by over 99%, and demonstrates stronger cross-lingual generalization. Such findings underscore the potential of our approach to promote social equality, ensure user privacy, and champion linguistic diversity.
- **OpenReview**: https://openreview.net/pdf?id=zzqn5G9fjn
        
</details>

### Enhancing Small Medical Learners with Privacy-preserving Contextual Prompting
> This method introduces an innovative approach to boost the performance of SLMs in the healthcare sector while simultaneously mitigating privacy concerns. By leveraging LLMs' medical knowledge to generate a specific context, SLMs can enhance their decision-making abilities and achieve performance comparable to LLMs even in privacy-restricted scenarios.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) demonstrate remarkable medical expertise, but data privacy concerns impede their direct use in healthcare environments. Although offering improved data privacy protection, domain-specific small language models (SLMs) often underperform LLMs, emphasizing the need for methods that reduce this performance gap while alleviating privacy concerns. In this paper, we present a simple yet effective method that harnesses LLMs' medical proficiency to boost SLM performance in medical tasks under $privacy-restricted$ scenarios. Specifically, we mitigate patient privacy issues by extracting keywords from medical data and prompting the LLM to generate a medical knowledge-intensive context by simulating clinicians' thought processes. This context serves as additional input for SLMs, augmenting their decision-making capabilities. Our method significantly enhances performance in both few-shot and full training settings across three medical knowledge-intensive tasks, achieving up to a 22.57% increase in absolute accuracy compared to SLM fine-tuning without context, and sets new state-of-the-art results in two medical tasks within privacy-restricted scenarios. Further out-of-domain testing and experiments in two general domain datasets showcase its generalizability and broad applicability.
- **OpenReview**: https://openreview.net/pdf?id=ztpy1gsUpT
        
</details>

### Detecting Pretraining Data from Large Language Models
> This paper addresses the challenge of detecting data used to train large language models (LLMs) by proposing a novel method called MIN-K PROB. Unlike existing methods that rely on training a reference model, MIN-K PROB identifies pretraining data based on the assumption that unseen examples contain outlier words with low probabilities under the LLM.

<details>
<summary>Details</summary>
- **Abstract**: Although large language models (LLMs) are widely deployed, the data used to train them is rarely disclosed. Given the incredible scale of this data, up to trillions of tokens, it is all but certain that it includes potentially problematic text such as copyrighted materials, personally identifiable information, and test data for widely reported reference benchmarks. However, we currently have no way to know which data of these types is included or in what proportions. In this paper, we study the pretraining data detection problem: given a piece of text and black-box access to an LLM without knowing the pretraining data, can we determine if the model was trained on the provided text? To facilitate this study, we introduce a dynamic benchmark WIKIMIA that uses data created before and after model training to support gold truth detection. We also introduce a new detection method MIN-K PROB based on a simple hypothesis: an unseen example is likely to contain a few outlier words with low probabilities under the LLM, while a seen example is less likely to have words with such low probabilities. MIN-K PROB can be applied without any knowledge about the pretrainig corpus or any additional training, departing from previous detection methods that require training a reference model on data that is similar to the pretraining data. Moreover, our experiments demonstrate that MIN-K PROB achieves a 7.4% improvement on WIKIMIA over these previous methods. We apply MIN-K PROB to two real-world scenarios, copyrighted book detection and contaminated downstream example detection, and find that it to be a consistently effective solution.
- **OpenReview**: https://openreview.net/pdf?id=zWqr3MQuNs
        
</details>

### AgentBench: Evaluating LLMs as Agents
> AgentBench is a collection of tasks that evaluates the reasoning and decision-making skills of large language models as agents in interactive environments, revealing a gap between commercial and open-source LLMs.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) are becoming increasingly smart and autonomous, targeting real-world pragmatic missions beyond traditional NLP tasks.  As a result, there has been an urgent need to evaluate LLMs as agents on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional evolving benchmark that currently consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities in a multi-turn open-ended generation setting. Our extensive test over 27 API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and OSS competitors. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Training on code and high quality multi-turn alignment data could improve agent performance. Datasets, environments, and an integrated evaluation package for AgentBench are released.
- **OpenReview**: https://openreview.net/pdf?id=zAdUB0aCTQ
        
</details>

### MathCoder: Seamless Code Integration in LLMs for Enhanced Mathematical Reasoning
> This paper introduces a novel method that enhances the mathematical reasoning capabilities of language models by fine-tuning them on a dataset that combines natural language, code, and execution results. The resulting models, called MathCoder, demonstrate impressive performance in solving challenging math problems, outperforming state-of-the-art open-source language models and even surpassing GPT-4 on a competition-level dataset.

<details>
<summary>Details</summary>
- **Abstract**: The recently released GPT-4 Code Interpreter has demonstrated remarkable proficiency in solving challenging math problems, primarily attributed to its ability to seamlessly reason with natural language, generate code, execute code, and continue reasoning based on the execution output. In this paper, we present a method to fine-tune open-source language models, enabling them to use code for modeling and deriving math equations and, consequently, enhancing their mathematical reasoning abilities. We propose a method of generating novel and high-quality datasets with math problems and their code-based solutions, referred to as MathCodeInstruct. Each solution interleaves $\\textit{natural language}$, $\\textit{code}$, and $\\textit{execution results}$. We also introduce a customized supervised fine-tuning and inference approach. This approach yields the MathCoder models, a family of models capable of generating code-based solutions for solving challenging math problems. Impressively, the MathCoder models achieve state-of-the-art scores among open-source LLMs on the MATH (45.2%) and GSM8K (83.9%) datasets, substantially outperforming other open-source alternatives. Notably, the MathCoder model not only surpasses ChatGPT-3.5 and PaLM-2 on GSM8K and MATH but also outperforms GPT-4 on the competition-level MATH dataset. The proposed dataset and models will be released upon acceptance.
- **OpenReview**: https://openreview.net/pdf?id=z8TW0ttBPp
        
</details>

### Achieving Fairness in Multi-Agent MDP Using Reinforcement Learning
> One can use Reinforcement Learning (RL) to achieve fairness in multi-agent systems with unknown environments by introducing a fairness function and making use of online convex optimization. This approach not only guarantees sub-linear regret, but also provides PAC guarantees and an offline RL algorithm with optimality bounds.

<details>
<summary>Details</summary>
- **Abstract**: Fairness plays a crucial role in various multi-agent systems (e.g., communication networks, financial markets, etc.). Many multi-agent dynamical interactions can be cast as Markov Decision Processes (MDPs). While existing research has focused on studying fairness in known environments, the exploration of fairness in such systems for unknown environments remains open. In this paper, we propose a  Reinforcement Learning (RL) approach to achieve fairness in multi-agent finite-horizon episodic MDPs. Instead of maximizing the sum of individual agents' value functions, we introduce a fairness function that ensures equitable rewards across agents. Since the classical Bellman's equation does not hold when the sum of individual value functions is not maximized, we cannot use traditional approaches. Instead, in order to explore, we maintain a confidence bound of the unknown environment and then propose an online convex optimization based approach to obtain a policy constrained to this confidence region. We show that such an approach achieves sub-linear regret in terms of the number of episodes. Additionally, we provide a probably approximately correct (PAC) guarantee based on the obtained regret bound. We also propose an offline RL algorithm and bound the optimality gap with respect to the optimal fair solution. To mitigate computational complexity, we introduce a policy-gradient type method for the fair objective. Simulation experiments also demonstrate the efficacy of our approach.
- **OpenReview**: https://openreview.net/pdf?id=yoVq2BGQdP
        
</details>

### Conversational Drug Editing Using Retrieval and Domain Feedback
> ChatDrug is an effective framework that empowers LLMs to perform drug editing tasks by offering prompts, retrieving relevant information, and engaging in conversations. Its superior performance and diverse suggestion generation make it valuable for identifying crucial substructures for manipulation and insightful explanations during drug discovery.

<details>
<summary>Details</summary>
- **Abstract**: Recent advancements in conversational large language models (LLMs), such as ChatGPT, have demonstrated remarkable promise in various domains, including drug discovery. However, existing works mainly focus on investigating the capabilities of conversational LLMs on chemical reactions and retrosynthesis. While drug editing, a critical task in the drug discovery pipeline, remains largely unexplored. To bridge this gap, we propose ChatDrug, a framework to facilitate the systematic investigation of drug editing using LLMs. ChatDrug jointly leverages a prompt module, a retrieval and domain feedback module, and a conversation module to streamline effective drug editing. We empirically show that ChatDrug reaches the best performance on all 39 drug editing tasks, encompassing small molecules, peptides, and proteins. We further demonstrate, through 10 case studies, that ChatDrug can successfully identify the key substructures for manipulation, generating diverse and valid suggestions for drug editing. Promisingly, we also show that ChatDrug can offer insightful explanations from a domain-specific perspective, enhancing interpretability and enabling informed decision-making.
- **OpenReview**: https://openreview.net/pdf?id=yRrPfKyJQ2
        
</details>

### MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning
> MAmmoTH, a series of large language models designed for math problem-solving, outperforms existing models due to its training on a unique dataset that combines chain-of-thought and program-of-thought rationales, covering diverse mathematical fields.

<details>
<summary>Details</summary>
- **Abstract**: We introduce MAmmoTH, a series of open-source large language models (LLMs) specifically tailored for general math problem-solving. The MAmmoTH models are trained on MathInstruct, our meticulously curated instruction tuning dataset. MathInstruct is compiled from 13 math datasets with intermediate rationales, six of which have rationales newly curated by us. It presents a unique hybrid of chain-of-thought (CoT) and program-of-thought (PoT) rationales, and also ensures extensive coverage of diverse fields in math. The hybrid of CoT and PoT not only unleashes the potential of tool use but also allows different thought processes for different math problems. As a result, the MAmmoTH series substantially outperform existing open-source models on nine mathematical reasoning datasets across all scales with an average accuracy gain between 16% and 32%. Remarkably, our MAmmoTH-7B model reaches 33% on MATH (a competition-level dataset), which exceeds the best open-source 7B model (WizardMath) by 23%, and the MAmmoTH-34B model achieves 44% accuracy on MATH, even surpassing GPT-4's CoT result. Our work underscores the importance of diverse problem coverage and the use of hybrid rationales in developing superior math generalist models.
- **OpenReview**: https://openreview.net/pdf?id=yLClGs770I
        
</details>

### Is Self-Repair a Silver Bullet for Code Generation?
> While large language models demonstrate impressive code generation abilities, their self-repair capabilities, where they debug and correct their own mistakes, are limited, especially when considering the costs associated with repair.

<details>
<summary>Details</summary>
- **Abstract**: Large language models have shown remarkable aptitude in code generation, but still struggle on challenging tasks. Self-repair---in which the model debugs and fixes mistakes in its own code---has recently become a popular way to boost performance in these settings. However, only very limited studies on how and when self-repair works effectively exist in the literature, and one might wonder to what extent a model is really capable of repairing mistakes in code which was originally generated by that very same model. In this paper, we analyze Code Llama, GPT-3.5 and GPT-4's ability to perform self-repair on problems taken from HumanEval or APPS, finding that when the cost of carrying out repair is taken into account gains are often modest, vary a lot between subsets of the data, and are sometimes not present at all. We hypothesize that this is because self-repair is bottlenecked by the model's ability to provide feedback on its own code; boosting the feedback with stronger models, we observe performance gains even in settings where the model does not benefit from self-repair. Furthermore, we observe that providing the model with feedback from human participants greatly benefits repair even for GPT-4, and we provide a brief qualitative analysis as to why.
- **OpenReview**: https://openreview.net/pdf?id=y0GJXRungR
        
</details>

### DreamLLM: Synergistic Multimodal Comprehension and Creation
> DreamLLM, a novel framework for Multimodal Large Language Models, presents a unique approach to multimodal comprehension and creation, enabling the generation of interleaved documents that effectively capture complex multimodal relationships. Through direct sampling in the raw multimodal space and fostering the generation of raw interleaved documents, DreamLLM achieves better learning synergy and free-form interleaved content generation.

<details>
<summary>Details</summary>
- **Abstract**: This paper presents DreamLLM, a learning framework that first achieves versatile Multimodal Large Language Models (MLLMs) empowered with frequently overlooked synergy between multimodal comprehension and creation. DreamLLM operates on two fundamental principles. The first focuses on the generative modeling of both language and image posteriors by direct sampling in the raw multimodal space. This approach circumvents the limitations and information loss inherent to external feature extractors like CLIP, and a more thorough multimodal understanding is obtained. Second, DreamLLM fosters the generation of raw, interleaved documents, modeling both text and image contents, along with unstructured layouts. This allows DreamLLM to learn all conditional, marginal, and joint multimodal distributions effectively. As a result, DreamLLM is the first MLLM capable of generating free-form interleaved content. Comprehensive experiments highlight DreamLLM's superior performance as a zero-shot multimodal generalist, reaping from the enhanced learning synergy. Anonymous project page: https://dreamllmpaper.github.io.
- **OpenReview**: https://openreview.net/pdf?id=y01KGvd9Bw
        
</details>

### Retrieval meets Long Context Large Language Models
> This study compares the performance of retrieval-augmented vs. long context window LLMs for downstream tasks, finding that retrieval-augmentation can be as effective as using longer context windows, and that combining both methods can further enhance performance.

<details>
<summary>Details</summary>
- **Abstract**: Extending the context window of large language models (LLMs) is getting popular recently, while the solution of augmenting LLMs with retrieval has existed for years. The natural questions are: i) Retrieval-augmentation versus long context window, which one is better for downstream tasks? ii) Can both methods be combined to get the best of both worlds? In this work, we answer these questions by studying both solutions using two state-of-the-art pretrained LLMs, i.e., a proprietary 43B GPT and LLaMA2-70B. Perhaps surprisingly, we find that shorter context window LLM with simple retrieval-augmentation at inference can perform close to longer context LLM finetuned via positional interpolation for question answering and query-based summarization tasks, while taking much less computation. More importantly, we demonstrate that retrieval can significantly improve the performance of LLMs regardless of their context window sizes. Our study provides general insights on the choice of retrieval-augmentation versus long context extension of LLM for practitioners.
- **OpenReview**: https://openreview.net/pdf?id=xw5nxFWMlo
        
</details>

### Teaching Language Models to Hallucinate Less with Synthetic Tasks
> Large language models' hallucination tendencies may be reduced via a novel synthetic training approach, SynTra, suggesting that synthetic task optimization can ameliorate this issue even in real-world applications.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) frequently hallucinate on abstractive summarization tasks such as document-based question-answering, meeting summarization, and clinical report generation, even though all necessary information is included in context. However, optimizing to make LLMs hallucinate less is challenging, as hallucination is hard to efficiently, cheaply, and reliably evaluate at each optimization step. In this work, we show that reducing hallucination on a _synthetic task_ can also reduce hallucination on real-world downstream tasks. Our method, SynTra, first designs a synthetic task where hallucinations are easy to elicit and measure. It next optimizes the LLM's system message via prefix tuning on the synthetic task, then uses the system message on realistic, hard-to-optimize tasks. Across three realistic abstractive summarization tasks, we reduce hallucination for two 13B-parameter LLMs using supervision signal from only a synthetic retrieval task. We also find that optimizing the system message rather than the model weights can be critical; fine-tuning the entire model on the synthetic task can counterintuitively _increase_ hallucination. Overall, SynTra demonstrates that the extra flexibility of working with synthetic data can help mitigate undesired behaviors in practice.
- **OpenReview**: https://openreview.net/pdf?id=xpw7V0P136
        
</details>

### Statistical Rejection Sampling Improves Preference Optimization
> To improve the alignment of language models with human preferences, the paper introduces Statistical Rejection Sampling Optimization (RSO), which combines elements from Sequence Likelihood Calibration (SLiC) and Direct Preference Optimization (DPO) to enhance preference modeling and utilizes rejection sampling to sample preference data from the target optimal policy, resulting in superior performance in alignment tasks.

<details>
<summary>Details</summary>
- **Abstract**: Improving the alignment of language models with human preferences remains an active research challenge. Previous approaches have primarily utilized online Reinforcement Learning from Human Feedback (RLHF). Recently, offline methods such as Sequence Likelihood Calibration (SLiC) and Direct Preference Optimization (DPO) have emerged as attractive alternatives, offering improvements in stability and scalability while maintaining competitive performance. SLiC refines its loss function using sequence pairs sampled from a supervised fine-tuned (SFT) policy, while DPO directly optimizes language models based on preference data, foregoing the need for a separate reward model. However, the maximum likelihood estimator (MLE) of the target optimal policy requires labeled preference pairs sampled from that policy. The absence of a reward model in DPO constrains its ability to sample preference pairs from the optimal policy. Meanwhile, SLiC can only sample preference pairs from the SFT policy. To address these limitations, we introduce a novel approach called Statistical Rejection Sampling Optimization (RSO) designed to source preference data from the target optimal policy using rejection sampling, enabling a more accurate estimation of the optimal policy. We also propose a unified framework that enhances the loss functions used in both SLiC and DPO from a preference modeling standpoint. Through extensive experiments across diverse tasks, we demonstrate that RSO consistently outperforms both SLiC and DPO as evaluated by both Large Language Models (LLMs) and human raters.
- **OpenReview**: https://openreview.net/pdf?id=xbjSwwrQOe
        
</details>

### Tell Your Model Where to Attend: Post-hoc Attention Steering for LLMs
> PASTA, a novel attention-steering approach, empowers users to guide large language models' attention by marking text with emphasis. Experiments show that PASTA enhances LLMs' instruction-following and knowledge integration capabilities, resulting in significant performance improvements on various tasks.

<details>
<summary>Details</summary>
- **Abstract**: In human-written articles, we often leverage the subtleties of text style, such as bold and italics, to guide the attention of readers. These textual emphases are vital for the readers to grasp the conveyed information.  When interacting with large language models (LLMs), we have a similar need -- steering the model to pay closer attention to user-specified information, e.g., an instruction. Existing methods, however, are constrained to process plain text and do not support such a mechanism. This motivates us to introduce PASTA -- Post-hoc Attention STeering Approach, a method that allows LLMs to read text with user-specified emphasis marks. To this end, PASTA identifies a small subset of attention heads and applies precise attention reweighting on them, directing the model attention to user-specified parts. Like prompting, PASTA is applied at inference time and does not require changing any model parameters. Experiments demonstrate that PASTA can substantially enhance an LLM's ability to follow user instructions or integrate new knowledge from user inputs, leading to a significant performance improvement on a variety of tasks, e.g., an average accuracy improvement of 22% for LLAMA-7B. Code is provided at https://anonymous.4open.science/r/PASTA-10E9.
- **OpenReview**: https://openreview.net/pdf?id=xZDWO0oejD
        
</details>

### Impact of Computation in Integral Reinforcement Learning for Continuous-Time Control
> IntRL's policy evaluation stage, the computation of the utility integral, is demonstrated to significantly impact control performance due to computational errors affecting policy iteration. Researchers present Bayesian quadrature in reproducing kernel Hilbert spaces as the optimal computational method, showing convergence rates for trapezoidal and Matern kernel-based Bayesian quadrature.

<details>
<summary>Details</summary>
- **Abstract**: Integral reinforcement learning (IntRL) demands the precise computation of the utility function's integral at its policy evaluation (PEV) stage. This is achieved through quadrature rules, which are weighted sums of utility functions evaluated from state samples obtained in discrete time. Our research reveals a critical yet underexplored phenomenon: the choice of the computational method -- in this case, the quadrature rule -- can significantly impact control performance. This impact is traced back to the fact that computational errors introduced in the PEV stage can affect the policy iteration's convergence behavior, which in turn affects the learned controller. To elucidate how computation impacts control, we draw a parallel between IntRL's policy iteration and Newton's method applied to the Hamilton-Jacobi-Bellman equation. In this light, computational error in PEV manifests as an extra error term in each iteration of Newton's method, with its upper bound proportional to the computational error. Further, we demonstrate that when the utility function resides in a reproducing kernel Hilbert space (RKHS), the optimal quadrature is achievable by employing Bayesian quadrature with the RKHS-inducing kernel function. We prove that the local convergence rates for IntRL using the trapezoidal rule and Bayesian quadrature with a Matern kernel to be $O(N^{-2})$ and $O(N^{-b})$, where $N$ is the number of evenly-spaced samples and $b$ is the Matern kernel's smoothness parameter. These theoretical findings are finally validated by two canonical control tasks.
- **OpenReview**: https://openreview.net/pdf?id=xJEd8PkdNz
        
</details>

### Privacy-Preserving In-Context Learning for Large Language Models
> In-context learning enhances the performance of Large Language Models, but can expose sensitive information. The paper introduces Differentially Private In-context Learning (DP-ICL), a technique for privatizing ICL tasks by generating noisy responses based on multiple, anonymized examples.

<details>
<summary>Details</summary>
- **Abstract**: In-context learning (ICL) is an important capability of Large Language Models (LLMs), enabling these models to dynamically adapt based on specific, in-context exemplars, thereby improving accuracy and relevance. However, LLM's responses may leak the sensitive private information contained in in-context exemplars.  To address this challenge, we propose Differentially Private In-context Learning (DP-ICL), a general paradigm for privatizing ICL tasks.  The key idea for DP-ICL paradigm is generating differentially private responses through a noisy consensus among an ensemble of LLM's responses based on disjoint exemplar sets.  Based on the general paradigm of DP-ICL, we instantiate several techniques showing how to privatize ICL for text classification and language generation.  We experiment on four text classification benchmarks and two language generation tasks, and our empirical findings suggest that our DP-ICL achieves a strong utility-privacy tradeoff.
- **OpenReview**: https://openreview.net/pdf?id=x4OPJ7lHVU
        
</details>

### Urial: Aligning Untuned LLMs with Just the 'Write' Amount of In-Context Learning"
> Alignment tuning methods like supervised fine-tuning and reinforcement learning from human feedback enhance LLMs, but it remains unclear what they learn during this process. Our study reveals that LLMs primarily acquire the language style of AI assistants during alignment tuning, raising the question of whether weight updates are essential for LLM alignment.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have shown significant improvements due to alignment tuning, that is, supervised fine-tuning (SFT) on instruction data and reinforcement learning from human feedback (RLHF). This raises questions about what is precisely learned during the alignment tuning process. We investigate the effects of alignment tuning through the lens of  token distribution shift between untuned LLMs and their aligned counterparts (e.g., Llama-2 versus Llama-2-Chat). Our findings reveal that most distribution changes lie in stylistic tokens (e.g., transitional words, discourse markers), suggesting that LLMs primarily learn the language style of AI assistants during alignment tuning, while most of useful knowledge has been acquired by untuned LLMs. Thus, we pose the question: Is it necessary to update model weights to attain LLM alignment? Based on these insights, we propose an alternative method, Untuned LLMs with Restyled In-context Alignment (\\textsc{Urial}), which achieves effective alignment solely through in-context learning (ICL) with as few as three curated, stylistic examples. Our evaluation on diverse examples from LIMA and AlpacaEval demonstrates that \\textsc{Urial} can achieve highly satisfactory performance, sometimes equaling or surpassing SFT+RLHF counterparts, especially when the untuned LLM is sufficiently pre-trained. This implies that fine-tuning may not be as always crucial as previously assumed for LLM alignment, and lightweight alignment methods like \\textsc{Urial} hold promise for efficiently tailoring LLM behavior without fine-tuning.
- **OpenReview**: https://openreview.net/pdf?id=wxJ0eXwwda
        
</details>

### CLEX: Continuous  Length Extrapolation for Large Language Models
> Continuous Length Extrapolation (CLEX) extends the limited context window of Transformer models by modeling continuous length scaling using differential equations, enabling effective length extrapolation beyond training sequence length. Despite its simplicity and compatibility with existing models, CLEX achieves impressive performance in long-context applications.

<details>
<summary>Details</summary>
- **Abstract**: Transformer-based Large Language Models (LLMs) are pioneering advances in many natural language processing tasks, however, their exceptional capabilities are restricted within the preset context window of Transformer. Position Embedding (PE) scaling methods, while effective in extending the context window to a specific length, demonstrate either notable limitations in their extrapolation abilities or sacrificing partial performance within the context window. Length extrapolation methods, although theoretically capable of extending the context window beyond the training sequence length, often underperform in practical long-context applications. To address these challenges, we propose Continuous Length EXtrapolation (CLEX) for LLMs. We generalise the PE scaling approaches to model the continuous dynamics by ordinary differential equations over the length scaling factor, thereby overcoming the constraints of current PE scaling methods designed for specific lengths. Moreover, by extending the dynamics to desired context lengths beyond the training sequence length, CLEX facilitates the length extrapolation with impressive performance in practical tasks. We demonstrate that CLEX can be seamlessly incorporated into LLMs equipped with Rotary Position Embedding, such as LLaMA and GPT-NeoX, with negligible impact on training and inference latency. Experimental results reveal that CLEX can effectively extend the context window to over 4x training length, with no deterioration in performance. Furthermore, when evaluated on the practical LongBench benchmark, our model trained on a 4k length exhibits competitive performance against state-of-the-art open-source models trained on context lengths up to 32k.
- **OpenReview**: https://openreview.net/pdf?id=wXpSidPpc5
        
</details>

### SuRe: Improving Open-domain Question Answering of LLMs via Summarized Retrieval
> This paper introduces SuRe, a framework that enhances open-domain question answering capabilities of large language models by incorporating summarized retrieval passages. SuRe enables LLMs to generate more grounded answers by leveraging summaries of retrieved passages, leading to significant improvements in question answering performance.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have made significant advancements in various natural language processing tasks but face challenges such as hallucinations and integration of up-to-date knowledge, which is particularly critical for question answering (QA). While incorporating new information with the retrieval of relevant passages is a promising way to improve QA with LLMs, the existing methods often require additional fine-tuning which becomes infeasible with recent LLMs. Retrieval augmentation via prompting has the potential to address this limitation, but this direction has been limitedly explored. To this end, we design a simple yet effective framework to enhance open-domain QA (ODQA) with LLMs, based on the summarized retrieval (SuRe). SuRe helps LLMs predict more grounded answers, which are well-supported by the summarization of retrieved passages that could be viewed as an explicit rationale extracted from the retrieved passages.  Specifically, SuRe first constructs summaries of the retrieved passages for each of the multiple answer candidates. Then, SuRe confirms the most plausible answer from the candidate set by evaluating the validity and ranking of the generated summaries. Experimental results on diverse ODQA benchmarks demonstrate the superiority of SuRe, with improvements of up to 4.4% in exact match (EM) and 3.9% in F1 score over standard prompting approaches. SuRe also can be integrated with a broad range of retrieval methods and LLMs. Finally, the generated summaries from SuRe show additional advantages to measure the importance of retrieved passages and serve as more preferred rationales by models and humans.
- **OpenReview**: https://openreview.net/pdf?id=w4DW6qkRmt
        
</details>

### Can Large Language Models Infer Causation from Correlation?
> Large language models struggle with causal inference, as demonstrated by a new dataset and task that evaluates their ability to determine causal relationships from correlational information, highlighting the need for improving their reasoning and generalization skills.

<details>
<summary>Details</summary>
- **Abstract**: Causal inference is one of the hallmarks of human intelligence. While the field of CausalNLP has attracted much interest in the recent years, existing causal inference datasets in NLP primarily rely on discovering causality from empirical knowledge (e.g. commonsense knowledge). In this work, we propose the first benchmark dataset to test the pure causal inference skills of large language models (LLMs). Specifically, we formulate a novel task Corr2Cause, which takes a set of correlational statements and determines the causal relationship between the variables. We curate a large-scale dataset of more than 400K samples, on which we evaluate seventeen existing LLMs. Through our experiments, we identify a key shortcoming of LLMs in terms of their causal inference skills, and show that these models achieve almost close to random performance on the task. This shortcoming is somewhat mitigated when we try to re-purpose LLMs for this skill via finetuning, but we find that these models still fail to generalize  they can only perform causal inference in in-distribution settings when variable names and textual expressions used in the queries are similar to those in the training set, but fail in out-of-distribution settings generated by perturbing these queries. Corr2Cause is a challenging task for LLMs, and would be helpful in guiding future research on improving LLMs pure reasoning ability and generalizability.
- **OpenReview**: https://openreview.net/pdf?id=vqIH0ObdqL
        
</details>

### CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules
> CodeChain enhances the code generation capabilities of LLMs by encouraging modularity and self-revision, leading to more efficient and accurate solutions for complex programming tasks.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have already become quite proficient at solving simpler programming tasks like those in HumanEval or MBPP benchmarks. However, solving more complex and competitive programming tasks is still quite challenging for these models - possibly due to their tendency to generate solutions as monolithic code blocks instead of decomposing them into logical sub-tasks and sub-modules. On the other hand, experienced programmers instinctively write modularized code with abstraction for solving complex tasks, often reusing previously developed modules. To address this gap, we propose CodeChain, a novel framework for inference that elicits modularized code generation through a chain of self-revisions, each being guided by some representative sub-modules generated in previous iterations. Concretely, CodeChain first instructs the LLM to generate modularized codes through chain-of-thought prompting. Then it applies a chain of self-revisions by iterating the two steps: 1) extracting and clustering the generated sub-modules and selecting the cluster representatives as the more generic and re-usable implementations, and 2) augmenting the original chain-of-thought prompt with these selected module-implementations and instructing the LLM to re-generate new modularized solutions. We find that by naturally encouraging the LLM to reuse the previously developed and verified sub-modules, CodeChain can significantly boost both modularity as well as correctness of the generated solutions, achieving relative pass@1 improvements of 35% on APPS and 76% on CodeContests. It is shown to be effective on both OpenAI LLMs as well as open-sourced LLMs like WizardCoder. We also conduct comprehensive ablation studies with different methods of prompting, number of clusters, model sizes, program qualities, etc., to provide useful insights that underpin CodeChain's success.
- **OpenReview**: https://openreview.net/pdf?id=vYhglxSj8j
        
</details>

### SliceGPT: Compress Large Language Models by Deleting Rows and Columns
> SliceGPT suggests a novel sparsification approach that alleviates resource constraints in large language models by reducing embedding dimensions without compromising accuracy, allowing for faster inference and reduced compute costs.

<details>
<summary>Details</summary>
- **Abstract**: Large language models have become the cornerstone of natural language processing, but their use comes with substantial costs in terms of compute and memory resources. Sparsification provides a solution to alleviate these resource constraints, and recent works have shown that trained models can be sparsified post-hoc. Existing sparsification techniques face challenges as they need additional data structures and offer constrained speedup with current hardware.  In this paper we present SliceGPT, a new post-training sparsification scheme which replaces each weight matrix with a smaller (dense) matrix, reducing the embedding dimension of the network. Through extensive experimentation, we show that SliceGPT can remove up to 25% of the model parameters (including embeddings) for  OPT 66B and LLAMA-2 70B models with negligible loss in accuracy. Our sliced models run on fewer GPUs and run faster without any additional code optimization: on 24GB consumer GPUs we reduce the total compute for inference on LLAMA-2 70B to 64% of that of the dense model; on 40GB A100 GPUs we reduce it to 66%. We offer a new insight, computational invariance in transformer networks, which enables SliceGPT and we hope it will inspire and enable future avenues to reduce memory and computation demands for pre-trained models.
- **OpenReview**: https://openreview.net/pdf?id=vXxardq6db
        
</details>

### Fine-Tuned Language Models Generate Stable Inorganic Materials as Text
> This paper proposes using fine-tuned language models to generate stable materials, achieving a higher rate of metastable materials generation (49%) than existing diffusion models (28%). The models can generate stable materials unconditionally, fill in partial structures, and perform text-conditional generation, leveraging the flexibility of text prompting.

<details>
<summary>Details</summary>
- **Abstract**: Deep learning models have drastically accelerated materials discovery by accelerating predictive computational simulations like density functional theory (DFT).  Large open computational materials databases such as the Materials Project or OQMD contain O($10^6$) known structures, and it is now straightforward to search those databases for materials with exciting properties. However, these databases are limited to experimentally known materials or candidates discovered in high-throughput computational campaigns. Many state-of-the-art engineering advances in solar photovaltaics, battery electrodes, and catalysts are made by discovering materials with outstanding properties that have not yet been discovered. Generative models are a natural solution to expand families of interest through sampling. While popular methods are typically constructed from variational autoencoders or diffusion models, we propose fine-tuning large language models for generation of stable materials. While unorthodox, fine-tuning large language models on text-encoded atomistic data is simple to implement yet reliable, with around 90% of sampled structures obeying physical constraints on atom positions and charges. Using energy of hull calculations from both learned ML potentials and gold-standard DFT calculations, we show that our strongest model (fine-tuned  LLaMA-2 70B) can generate materials predicted to be metastable at about twice the rate (49% vs 28%) of CDVAE, a competing diffusion model. Because of text prompting's inherent flexibility, our models can simultaneously be used for unconditional generation of stable material, infilling of partial structures and text-conditional generation. Finally, we show that language models' ability to capture key symmetries of crystal structures improves with model scale, suggesting that the biases of pretrained LLMs are surprisingly well-suited for atomistic data.
- **OpenReview**: https://openreview.net/pdf?id=vN9fpfqoP1
        
</details>

### Multilingual Jailbreak Challenges in Large Language Models
> This study highlights the multilingual dimensions of jailbreaking risks in large language models, emphasizing the need for safety measures in multiple languages. The findings reveal the increased susceptibility of low-resource languages to harmful content and the potential for malicious actors to exploit multilingual prompts.

<details>
<summary>Details</summary>
- **Abstract**: While large language models (LLMs) exhibit remarkable capabilities across a wide range of tasks, they pose potential safety concerns, such as the ``jailbreak'' problem. Although several preventive measures have been developed to mitigate the potential risks associated with LLMs, they have primarily focused on English data. In this study, we reveal the presence of multilingual jailbreak challenges within LLMs and consider two potential risky scenarios: unintentional and intentional. The unintentional scenario involves users querying LLMs using non-English prompts and inadvertently bypassing the safety mechanisms, while the intentional scenario entails malicious users combining jailbreak instructions with multilingual prompts to attack LLMs deliberately. The experimental results reveal that in the unintentional scenario, the rate of unsafe content increases as the availability of languages decreases. Specifically, low-resource languages exhibit three times the likelihood of encountering harmful content compared to high-resource languages, with both ChatGPT and GPT-4. In the intentional scenario, multilingual prompts can exacerbate the negative impact of jailbreak instructions, with astonishingly high rates of unsafe output: 80.92% for ChatGPT and 40.71% for GPT-4. Finally, we propose a novel \\textsc{Self-Defense} framework that addresses the  multilingual jailbreak challenges  via automatically generating multilingual safety training data for fine-tuning. Experiment results demonstrate its effectiveness with notable reduction in unsafe rate.
- **OpenReview**: https://openreview.net/pdf?id=vESNKdEMGp
        
</details>

### Hierarchical Context Merging: Better Long Context Understanding for Pre-trained LLMs
> HOMER, a training-free approach, overcomes the context length limitation of large language models (LLMs) by segmenting inputs and using a hierarchical fusion strategy, reducing memory requirement and allowing LLMs to handle extended contexts.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have established new standards in various natural language processing tasks.  However, a primary constraint they face is the context limit, i.e., the maximum number of tokens they can process. To relax the constraint, previous works have explored architectural changes and modifications in positional encoding, but they often require expensive training or do not address the computational demands of self-attention. In this paper, we present Hierarchical cOntext MERging (HOMER), a new training-free scheme designed to overcome the limitations. HOMER harnesses a divide-and-conquer methodology, segmenting extensive inputs into manageable units. The segments are then processed collectively, employing a hierarchical strategy that fuses adjacent chunks at progressive Transformer layers. A token reduction technique precedes each fusion, ensuring memory usage efficiency. We also propose an optimized computational order reducing the memory requirement to logarithmically scale with respect to input length, making it especially favorable for environments with tight memory restrictions.  Our experimental results demonstrate the superior performance and memory efficiency of the proposed method, opening doors for broader applications of LLMs in scenarios with extended context requirements.
- **OpenReview**: https://openreview.net/pdf?id=ulaUJFd96G
        
</details>

### Unbiased Watermark for Large Language Models
> This study proposes `unbiased watermarks` for LLMs, which allow for tracking and attribution of model outputs without affecting the quality of the generated text, preserving the model's performance and utility, and fostering responsible AI development.

<details>
<summary>Details</summary>
- **Abstract**: The recent advancements in large language models (LLMs) have sparked a growing apprehension regarding the potential misuse. One approach to mitigating this risk is to incorporate watermarking techniques into LLMs, allowing for the tracking and attribution of model outputs. This study examines a crucial aspect of watermarking: how significantly watermarks impact the quality of model-generated outputs. Previous studies have suggested a trade-off between watermark strength and output quality. However, our research demonstrates that it is possible to integrate watermarks without affecting the output probability distribution with appropriate implementation. We refer to this type of watermark as an unbiased watermark. This has significant implications for the use of LLMs, as it becomes impossible for users to discern whether a service provider has incorporated watermarks or not. Furthermore, the presence of watermarks does not compromise the performance of the model in downstream tasks, ensuring that the overall utility of the language model is preserved. Our findings contribute to the ongoing discussion around responsible AI development, suggesting that unbiased watermarks can serve as an effective means of tracking and attributing model outputs without sacrificing output quality.
- **OpenReview**: https://openreview.net/pdf?id=uWVC5FVidc
        
</details>

### In-context Autoencoder for Context Compression in a Large Language Model
> ICAE, an innovative method, leverages language models to compress long text into concise memory slots for efficient processing, opening up new avenues for addressing challenges in working memory and context management in LLMs.

<details>
<summary>Details</summary>
- **Abstract**: We propose the In-context Autoencoder (ICAE), leveraging the power of a large language models (LLM) to compress a long context into short compact memory slots that can be directly conditioned on by the LLM for various purposes. ICAE is first pretrained using both autoencoding and language modeling objectives on massive text data, enabling it to generate memory slots that accurately and comprehensively represent the original context; Then, it is fine-tuned on instruction data for producing desirable responses to various prompts. Experiments demonstrate that our lightweight ICAE, introducing fewer than 1% additional parameters, effectively achieves $4\\times$ context compression based on Llama, offering advantages in both improved latency and GPU memory cost during inference, and showing an interesting insight in memorization as well as potential for scalability. These promising results imply a novel perspective on the connection between working memory in cognitive science and representation learning in LLMs, revealing ICAE's significant implications in addressing the long context problem and suggesting further research in LLM context management. Our data, code and model will be released.
- **OpenReview**: https://openreview.net/pdf?id=uREj4ZuGJE
        
</details>

### Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs
> This paper presents FastGen, an adaptive KV cache compression technique that significantly reduces memory usage in generative language models (LLMs) without compromising generation quality.

<details>
<summary>Details</summary>
- **Abstract**: In this study, we introduce adaptive KV cache compression, a plug-and-play method that reduces the memory footprint of generative inference for Large Language Models (LLMs). Different from the conventional KV cache that retains key and value vectors for all context tokens, we conduct targeted profiling to discern the intrinsic structure of attention modules. Based on the recognized structure, we then construct the KV cache in an adaptive manner: evicting long-range contexts on attention heads emphasizing local contexts, discarding non-special tokens on attention heads centered on special tokens, and only employing the standard KV cache for attention heads that broadly attend to all tokens. Moreover, with the lightweight attention profiling used to guide the construction of the adaptive KV cache, FastGen can be deployed without resource-intensive fine-tuning or re-training. In our experiments across various asks, FastGen demonstrates substantial reduction on GPU memory consumption with negligible generation quality loss. We will release our code and the compatible CUDA kernel for reproducibility.
- **OpenReview**: https://openreview.net/pdf?id=uNrFpDPMyo
        
</details>

### Entity-Centric Reinforcement Learning for Object Manipulation from Pixels
> In this paper, the authors propose an approach to enable Reinforcement Learning (RL) agents to manipulate multiple objects. Their approach incorporates a structured representation of objects and their interactions, allowing the agent to learn goal-conditioned manipulation tasks with object dependencies.

<details>
<summary>Details</summary>
- **Abstract**: Manipulating objects is a hallmark of human intelligence, and an important task in domains such as robotics. In principle, Reinforcement Learning (RL) offers a general approach to learn object manipulation. In practice, however, domains with more than a few objects are difficult for RL agents due to the curse of dimensionality, especially when learning from raw image observations. In this work we propose a structured approach for visual RL that is suitable for representing multiple objects and their interaction, and use it to learn goal-conditioned manipulation of several objects. Key to our method is the ability to handle goals with dependencies between the objects (e.g., moving objects in a certain order). We further relate our architecture to the generalization capability of the trained agent, and demonstrate agents that learn with 3 objects but generalize to similar tasks with over 10 objects. Rollout videos are available on our website: https://sites.google.com/view/entity-centric-rl'
- **OpenReview**: https://openreview.net/pdf?id=uDxeSZ1wdI
        
</details>

### Large Language Models as Generalizable Policies for Embodied Tasks
> LLaRP, a novel approach that utilizes LLMs as adaptable policies for embodied visual tasks, demonstrates the potential of LLMs to generalize to unseen tasks, including those requiring novel optimal behavior.

<details>
<summary>Details</summary>
- **Abstract**: We show that large language models (LLMs) can be adapted to be generalizable policies for embodied visual tasks. Our approach, called Large LAnguage model Reinforcement Learning Policy (LLaRP), adapts a pre-trained frozen LLM to take as input text instructions and visual egocentric observations and output actions directly in the environment. Using reinforcement learning, we train LLaRP to see and act solely through environmental interactions. We show that LLaRP is robust to complex paraphrasings of task instructions and can generalize to new tasks that require novel optimal behavior. In particular, on 1,000 unseen tasks it achieves 42% success rate, 1.7x the success rate of other common learned baselines or zero-shot applications of LLMs. Finally, to aid the community in studying language conditioned, massively multi-task, embodied AI problems we release a novel benchmark, Language Rearrangement, consisting of 150,000 training and 1,000 testing tasks for language-conditioned rearrangement.
- **OpenReview**: https://openreview.net/pdf?id=u6imHU4Ebu
        
</details>

### Evaluating Large Language Models at Evaluating Instruction Following
> The study evaluates the effectiveness of using one large language model (LLM) to assess the outputs of another LLM, specifically in the context of instruction following. The research team developed a challenging benchmark to test the ability of LLM evaluators in this task, revealing areas for improvement and introducing new prompting strategies to bridge the gap between LLM and human assessors.

<details>
<summary>Details</summary>
- **Abstract**: As research in large language models (LLMs) continues to accelerate, LLM-based evaluation has emerged as a scalable and cost-effective alternative to human evaluations for comparing the ever-increasing list of models. This paper investigates the efficacy of these LLM evaluators, particularly in using them to assess instruction following, a metric that gauges how closely generated text adheres to the instructions. We introduce a challenging meta-evaluation benchmark, LLMBAR, designed to test the ability of an LLM evaluator to discern instruction-following outputs. The authors curated 419 pairs of outputs, one adhering to instructions while the other diverging, yet may possess deceptive qualities that could mislead an LLM evaluator. Contrary to existing meta-evaluation, we discover that different evaluators (i.e., combinations of LLMs and prompts) exhibit distinct performance on LLMBAR and even the highest-scoring LLM evaluators have substantial room for improvement. We also present a novel suite of prompting strategies that further close the gap between LLM and human evaluators. With LLMBAR, we hope to offer more insight into the behavior of LLM evaluators and foster research in developing better instruction-following models.
- **OpenReview**: https://openreview.net/pdf?id=tr0KidwPLc
        
</details>

### Dissecting learning and forgetting in language model finetuning
> Finetuning language models on specific corpora improves domain performance but may lead to a loss of general knowledge. This study isolates the effects of finetuning on topic, style, and factual knowledge, revealing that while topic and style adaptations are rapid and superficial, factual knowledge acquisition is gradual and resource-intensive, providing insights into the complexities of language model adaptation and forgetting.

<details>
<summary>Details</summary>
- **Abstract**: Finetuning language models on domain-specific corpus is a common approach to enhance their domain knowledge and capability. While improving performance on domain tasks, it often brings a side-effect of forgetting of the model's general abilities. In this study, we analyze the effects of finetuning on language models by dissecting its impacts on the modeling of topic, style, and factual knowledge in text. Our method uses instruction-following LLMs such as ChatGPT to auto-generate controlled-variable text examples which we use to probe the model. Our findings reveal that finetuning results in significant shifts in the language model's topic and style priors, while actual knowledge learning only contributes to a small fraction of the total probability change. Analysis shows that the adaptation of topic and style priors behave akin to learning simple features: they are learned rapidly and require little model capacity. They are also learned independently and primarily at the beginning of a text sequence. In contrast, factual knowledge is learned stably but slowly and requires significant model capacity to learn. The research offers insights and understanding into the finer dynamics of learning and forgetting in language models, and can potentially inform future research on improving domain adaptation and addressing the challenges of forgetting in continual learning of language models.
- **OpenReview**: https://openreview.net/pdf?id=tmsqb6WpLz
        
</details>

### Motif: Intrinsic Motivation from Artificial Intelligence Feedback
> Motif proposes a way to integrate prior knowledge from language models into agent decision-making, leading to superior performance in challenging environments such as the NetHack game by aligning the agent's behavior with human intuition and facilitating adaptability through prompt modifications.

<details>
<summary>Details</summary>
- **Abstract**: Exploring rich environments and evaluating one's actions without prior knowledge is immensely challenging. In this paper, we propose Motif, a general method to interface such prior knowledge from a Large Language Model (LLM) with an agent. Motif is based on the idea of grounding LLMs for decision-making without requiring them to interact with the environment: it elicits preferences from an LLM over pairs of captions to construct an intrinsic reward, which is then used to train agents with reinforcement learning. We evaluate Motif's performance and behavior on the challenging, open-ended and procedurally-generated NetHack game. Surprisingly, by only learning to maximize its intrinsic reward, Motif achieves a higher game score than an algorithm directly trained to maximize the score itself. When combining Motif's intrinsic reward with the environment reward, our method significantly outperforms existing approaches and makes progress on tasks where no advancements have ever been made without demonstrations. Finally, we show that Motif mostly generates intuitive human-aligned behaviors which can be steered easily through prompt modifications, while scaling well with the LLM size and the amount of information given in the prompt.
- **OpenReview**: https://openreview.net/pdf?id=tmBKIecDE9
        
</details>

### A Benchmark for Learning to Translate a New Language from One Grammar Book
> LLMs face the challenge of learning new tasks with minimal data, as exemplified by the MTOB benchmark, where models translate between English and Kalamang using only a book of grammar explanations, demonstrating capabilities beyond internet-scale training sets and potential for language technology accessibility.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) can perform impressive feats with in-context learning or lightweight finetuning. It is natural to wonder how well these models adapt to genuinely new tasks, but how does one find tasks that are unseen in internet-scale training sets? We turn to a field that is explicitly motivated and bottlenecked by a scarcity of web data: low-resource languages. In this paper, we introduce MTOB (Machine Translation from One Book), a benchmark for learning to translate between English and Kalamanga language with less than 200 speakers and therefore virtually no presence on the webusing several hundred pages of field linguistics reference materials. This task framing is novel in that it asks a model to learn a language from a single human-readable book of grammar explanations, rather than a large mined corpus of in-domain data, more akin to L2 language learning than L1 language acquisition. We demonstrate that baselines using current LLMs are promising but fall short of human performance, achieving 44.7 chrF on Kalamang to English translation and 45.8 chrF on English to Kalamang translation, compared to 51.6 and 57.0 chrF by a human who learned Kalamang from the same reference materials. We hope that MTOB will help measure LLM capabilities along a new dimension, and that the methods developed to solve it could help expand access to language technology for underserved communities by leveraging qualitatively different kinds of data than traditional machine translation.
- **OpenReview**: https://openreview.net/pdf?id=tbVWug9f2h
        
</details>

### GraphCare: Enhancing Healthcare Predictions with Personalized Knowledge Graphs
> GraphCare, leveraging external knowledge graphs and large language models, advances healthcare predictions by constructing personalized knowledge graphs for each patient, leading to improved outcomes in mortality, readmission, hospital stay, and drug recommendations.

<details>
<summary>Details</summary>
- **Abstract**: Clinical predictive models often rely on patients electronic health records (EHR), but integrating medical knowledge to enhance predictions and decision-making is challenging. This is because personalized predictions require personalized knowledge graphs (KGs), which are difficult to generate from patient EHR data. To address this, we propose GraphCare, an open-world framework that uses external KGs to improve EHR-based predictions. Our method extracts knowledge from large language models (LLMs) and external biomedical KGs to build patient-specific KGs, which are then used to train our proposed Bi-attention AugmenTed (BAT) graph neural network (GNN) for healthcare predictions. On two public datasets, MIMIC-III and MIMIC-IV, GraphCare surpasses baselines in four vital healthcare prediction tasks: mortality, readmission, length of stay (LOS), and drug recommendation. On MIMIC-III, it boosts AUROC by 17.6% and 6.6% for mortality and readmission, and F1-score by 7.9% and 10.8% for LOS and drug recommendation, respectively. Notably, GraphCare demonstrates a substantial edge in scenarios with limited data availability. Our findings highlight the potential of using external KGs in healthcare prediction tasks and demonstrate the promise of GraphCare in generating personalized KGs for promoting personalized medicine.
- **OpenReview**: https://openreview.net/pdf?id=tVTN7Zs0ml
        
</details>

### Text2Reward: Dense Reward Generation with Language Models for Reinforcement Learning
> Text2Reward is a data-free framework that automates the generation of dense reward functions for RL, enabling users to describe goals in natural language and obtain executable programs grounded in the environment. It has demonstrated promising results on robotic manipulation and locomotion tasks, with the ability to learn novel behaviors and facilitate policy refinement through human feedback.

<details>
<summary>Details</summary>
- **Abstract**: Designing reward functions is a longstanding challenge in reinforcement learning (RL); it requires specialized knowledge or domain data, leading to high costs for development. To address this, we introduce Text2Reward, a data-free framework that automates the generation of dense reward functions based on large language models (LLMs). Given a goal described in natural language, Text2Reward generates dense reward functions as an executable program grounded in a compact representation of the environment. Unlike inverse RL and recent work that uses LLMs to write sparse reward codes, Text2Reward produces interpretable, free-form dense reward codes that cover a wide range of tasks, utilize existing packages, and allow iterative refinement with human feedback. We evaluate Text2Reward on two robotic manipulation benchmarks (Maniskill2, MetaWorld) and two locomotion environments of MuJoCo. On 13 of the 17 manipulation tasks, policies trained with generated reward codes achieve similar or better task success rates and convergence speed than expert-written reward codes. For locomotion tasks, our method learns six novel locomotion behaviors with a success rate exceeding 94%. Furthermore, we show that the policies trained in the simulator with our method can be deployed in the real world. Finally, Text2Reward further improves the policies by refining their reward functions with human feedback. Video results are available at https://text-to-reward-review.github.io.
- **OpenReview**: https://openreview.net/pdf?id=tUM39YTRxH
        
</details>

### Tailoring Self-Rationalizers with Multi-Reward Distillation
> Small-scale LMs can now generate rationales for question answering that are plausible, diverse, and consistent by incorporating multiple rewards into their training procedure, improving task accuracy and rationale quality.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LMs) are capable of generating free-text rationales to aid question answering. However, prior work 1) suggests that useful self-rationalization is emergent only at significant scales (e.g., 175B parameter GPT-3); and 2) focuses largely on downstream performance, ignoring the semantics of the rationales themselves, e.g., are they faithful, true, and helpful for humans? In this work, we enable small-scale LMs (200x smaller than GPT-3) to generate rationales that not only improve downstream task performance, but are also more plausible, consistent, and diverse, assessed both by automatic and human evaluation. Our method, MaRio (Multi-rewArd RatIOnalization), is a multi-reward conditioned self-rationalization algorithm that optimizes multiple distinct properties like plausibility, diversity and consistency. Results on three difficult question-answering datasets StrategyQA, QuaRel and OpenBookQA show that not only does MaRio improve task accuracy, but it also improves the self-rationalization quality of small LMs across the aforementioned axes better than a supervised fine-tuning (SFT) baseline. Extensive human evaluations confirm that MaRio rationales are preferred vs. SFT rationales, as well as qualitative improvements in plausibility and consistency.
- **OpenReview**: https://openreview.net/pdf?id=t8eO0CiZJV
        
</details>

### Frozen Transformers in Language Models Are Effective Visual Encoder Layers
> Large language models (LLMs) excel at visual tasks even without language input, by employing pre-trained transformer blocks to directly process visual information. This finding expands the capabilities of LLMs beyond language, while the information filtering hypothesis suggests that LLMs enhance visual recognition by focusing on relevant regions.

<details>
<summary>Details</summary>
- **Abstract**: This paper reveals that large language models (LLMs), despite being trained solely on text data, are \\emph{surprisingly} strong encoders for \\emph{purely} visual tasks in the absence of language. Even more intriguingly, this can be achieved by a simple yet previously overlooked strategy -- employing a \\emph{frozen} transformer block from \\emph{pre-trained} LLMs as a constituent encoder layer to directly process visual tokens. Our work pushes the boundaries of leveraging LLMs for computer vision tasks, significantly departing from conventional practices that typically necessitate a multi-modal vision-language setup with associated language prompts, inputs, or outputs. We demonstrate that our approach consistently enhances performance across \\emph{a diverse range of tasks}, encompassing pure 2D or 3D visual recognition tasks (e.g., image and point cloud classification), temporal modeling tasks (e.g., action recognition), non-semantic tasks (e.g., motion forecasting), and multi-modal tasks (e.g., 2D/3D visual question answering and image-text retrieval). Such improvements are a general phenomenon, applicable to various types of LLMs (e.g., LLaMA and OPT) and different LLM transformer blocks. We additionally propose the \\emph{information filtering} hypothesis to explain the effectiveness of pre-trained LLMs in visual encoding -- the pre-trained LLM transformer blocks discern informative visual tokens and further amplify their effect. This hypothesis is empirically supported by the observation that the feature activation, after training with LLM transformer blocks, exhibits a stronger focus on relevant regions. We hope that our work inspires new perspectives on utilizing LLMs and deepening our understanding of their underlying mechanisms.
- **OpenReview**: https://openreview.net/pdf?id=t0FI3Q66K5
        
</details>

### Large Language Models Are Not Robust Multiple Choice Selectors
> LLMs are prone to answering MCQs based on option position (e.g., "Option A") due to their inherent preference for certain answer IDs. PriDe, a novel method, addresses this selection bias by separating the model\'s option ID preference from the answer distribution, resulting in enhanced debiasing performance and interpretable insights into the model\'s behavior.

<details>
<summary>Details</summary>
- **Abstract**: Multiple choice questions (MCQs) serve as a common yet important task format in the research of large language models (LLMs). This work shows that LLMs are vulnerable to option position changes in MCQs due to their inherent selection bias, namely, they prefer to select specific option IDs as answers (like Option A). Through extensive empirical analyses with 20 LLMs on three benchmarks, we pinpoint that this behavioral bias primarily stems from LLMs token bias, where the model a priori assigns more probabilistic mass to specific option ID tokens (e.g., A/B/C/D) when predicting answers from the option IDs. To mitigate selection bias, we propose a label-free, inference-time debiasing method, called PriDe, which separates the models prior bias for option IDs from the overall prediction distribution. PriDe first estimates the prior by permutating option contents on a small number of test samples, which is then applied to debias the subsequent samples. We demonstrate that PriDe achieves superior debiasing effectiveness and computational efficiency to strong baselines. Furthermore, the prior estimated by PriDe is interpretable and can generalize well across different domains, highlighting its practical potential in broader scenarios.
- **OpenReview**: https://openreview.net/pdf?id=shr9PXz7T0
        
</details>

### Let Models Speak Ciphers: Multiagent Debate through Embeddings
> CIPHER, a novel communication method for LLMs, bypasses the limiting token sampling step, allowing them to exchange beliefs across the full vocabulary through transformer output embeddings. This deviation from natural language enables CIPHER to encode a broader information range, surpassing existing LLM debate methods and demonstrating the power of embeddings as an alternative communication language.

<details>
<summary>Details</summary>
- **Abstract**: Discussion and debate among Large Language Models (LLMs) have gained considerable attention due to their potential to enhance the reasoning ability of LLMs. Although natural language is an obvious choice for communication due to LLM\'s language understanding capability, the token sampling step needed when generating natural language poses a potential risk of information loss, as it uses only one token to represent the model\'s belief across the entire vocabulary. In this paper, we introduce a communication regime named CIPHER (Communicative Inter-Model Protocol Through Embedding Representation) to address this issue. Specifically, we remove the token sampling step from LLMs and let them communicate their beliefs across the vocabulary through the expectation of the raw transformer output embeddings. Remarkably, by deviating from natural language, CIPHER offers an advantage of encoding a broader spectrum of information without any modification to the model weights, outperforming the state-of-the-art LLM debate methods using natural language by 1-3.5% across five reasoning tasks and multiple open-source LLMs of varying sizes. This showcases the superiority and robustness of embeddings as an alternative ``language" for communication among LLMs.
- **OpenReview**: https://openreview.net/pdf?id=sehRvaIPQQ
        
</details>

### Octavius: Mitigating Task Interference in MLLMs via MoE
> Octavius, a proposed framework, combines Mixture-of-Experts with LoRA to mitigate negative interference in multimodal learning with Multimodal Large Language Models (MLLMs), enhancing task- and modality-specific performance by approximately 20%.

<details>
<summary>Details</summary>
- **Abstract**: Recent studies have demonstrated Large Language Models (LLMs) can extend their zero-shot generalization capabilities to multimodal learning through instruction tuning. As more modalities and downstream tasks are introduced, negative conflicts and interference may have a worse impact on performance. While this phenomenon has been overlooked in previous work, we propose a novel and extensible framework, called Octavius, for comprehensive studies and experimentation on multimodal learning with Multimodal Large Language Models (MLLMs). Specifically, to mitigate the interference, we combine the concept of Mixture-of-Experts (MoE) with LoRA and design a multimodal LoRA-MoE decoder for task- and modality-specific learning. The experimental results (about 20% improvement) have shown the effectiveness and versatility of our design in various 2D and 3D downstream tasks. Code and corresponding dataset will be available soon.
- **OpenReview**: https://openreview.net/pdf?id=rTDyN8yajn
        
</details>

### UniversalNER: Targeted Distillation from Large Language Models for Open Named Entity Recognition
> Can a large language model be refined to outperform itself in a specific task without compromising its versatility? This paper presents a targeted distillation technique that addresses this question by using mission-focused instructions to train specialized student models. The resulting UniversalNER model, distilled from ChatGPT, exhibits superior performance in open named entity recognition across diverse domains with significantly fewer parameters, even outperforming supervised multi-task systems.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have demonstrated remarkable generalizability, such as understanding arbitrary entities and relations. Instruction tuning has proven effective for distilling LLMs into more cost-efficient models such as Alpaca and Vicuna. Yet such student models still trail the original LLMs by large margins in downstream applications. In this paper, we explore targeted distillation with mission-focused instruction tuning to train student models that can excel in a broad application class such as open information extraction. Using named entity recognition (NER) for case study, we show how ChatGPT can be distilled into much smaller UniversalNER models for open NER. For evaluation, we assemble the largest NER benchmark to date, comprising 43 datasets across 9 diverse domains such as biomedicine, programming, social media, law, finance. Without using any direct supervision, UniversalNER attains remarkable NER accuracy across tens of thousands of entity types, outperforming general instruction-tuned models such as Alpaca and Vicuna by over 30 absolute F1 points in average. With a tiny fraction of parameters, UniversalNER not only acquires ChatGPT's capability in recognizing arbitrary entity types, but also outperforms its NER accuracy by 7-9 absolute F1 points in average. Remarkably, UniversalNER even outperforms by a large margin state-of-the-art multi-task instruction-tuned systems such as InstructUIE, which uses supervised NER examples. We also conduct thorough ablation studies to assess the impact of various components in our distillation approach. We will release the distillation recipe, data, and UniversalNER models to facilitate future research on targeted distillation.
- **OpenReview**: https://openreview.net/pdf?id=r65xfUb76p
        
</details>

### Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation
> This study highlights an inadequacy in current safety evaluations for open-source LLMs, arguing that malicious manipulations can impair model alignment, even with prior safeguards. The "generation exploitation" attack, which manipulates decoding methods, dramatically increases misalignment and outperforms previous attacks, suggesting the need for comprehensive security assessments and alignment enhancements before model release.

<details>
<summary>Details</summary>
- **Abstract**: The rapid progress in open-source large language models (LLMs) is significantly advancing AI development. Extensive efforts have been made before model release to align their behavior with human values, with the primary goal of ensuring their helpfulness and harmlessness. However, even carefully aligned models can be manipulated maliciously, leading to unintended behaviors, known as "jailbreaks". These jailbreaks are typically triggered by specific text inputs, often referred to as adversarial prompts. In this work, we propose the \\emph{generation exploitation} attack, an extremely simple approach that disrupts model alignment by only manipulating variations of decoding methods. By exploiting different generation strategies, including varying decoding hyper-parameters and sampling methods, we increase the misalignment rate from $0\\%$ to more than $95\\%$ across 11 language models including LLaMA2, Vicuna, Falcon, and MPT families, outperforming state-of-the-art attacks with $30\\times$ lower computational cost. Finally, we propose an effective alignment method that explores diverse generation strategies, which can reasonably reduce the misalignment rate under our attack. Altogether, our study underscores a major failure in current safety evaluation and alignment procedures for open-source LLMs, strongly advocating for more comprehensive red teaming and better alignment before releasing such models.
- **OpenReview**: https://openreview.net/pdf?id=r42tSSCHPh
        
</details>

### Demystifying Embedding Spaces using Large Language Models
> This research proposes using large language models (LLMs) to enhance the interpretability of embeddings, allowing for direct interaction and transformation of abstract vectors into comprehensible narratives, thus expanding their applicability to various tasks.

<details>
<summary>Details</summary>
- **Abstract**: Embeddings have become a pivotal means to represent complex, multi-faceted information about entities, concepts, and relationships in a condensed and useful format. Nevertheless, they often preclude direct interpretation. While downstream tasks make use of these compressed representations, meaningful interpretation usually requires visualization using dimensionality reduction or specialized machine learning interpretability methods. This paper addresses the challenge of making such embeddings more interpretable and broadly useful, by employing large language models (LLMs) to directly interact with embeddings -- transforming abstract vectors into understandable narratives. By injecting embeddings into LLMs, we enable querying and exploration of complex embedding data. We demonstrate our approach on a variety of diverse tasks, including: enhancing concept activation vectors (CAVs), communicating novel embedded entities, and decoding user preferences in recommender systems. Our work couples the immense information potential of embeddings with the interpretative power of LLMs.
- **OpenReview**: https://openreview.net/pdf?id=qoYogklIPz
        
</details>

### Grounding Language Plans in Demonstrations Through Counter-Factual Perturbations
> This paper explores the potential of Large Language Models (LLMs) to improve robot manipulation by leveraging concepts from plan-ning literature. Specifically, the authors introduce a framework that uses LLMs to ground abstract language representations into low-level physical trajectories, enabling the learning of structured policies for manipulation tasks.

<details>
<summary>Details</summary>
- **Abstract**: Grounding the abstract knowledge captured by Large Language Models (LLMs) in physical domains remains a pivotal yet unsolved problem. Whereas prior works have largely focused on leveraging LLMs for generating abstract plans in symbolic spaces, this work uses LLMs to guide the learning for structures and constraints in robot manipulation tasks. Specifically, we borrow from manipulation plan- ning literature the concept of mode families, defining specific types of motion constraints among sets of objects, to serve as an intermediate layer that connects high-level language representations with low-level physical trajectories. By lo- cally perturbing a small set of successful human demonstrations, we augment the dataset with additional successful executions as well as counterfactuals that fail the task. Our explanation-based learning framework trains neural network-based classifiers to differentiate success task executions from failures and as a by-product learns classifiers that ground low-level states into mode families without dense labeling. This further enables us to learn structured policies for the target task. Experimental validation in both 2D continuous-space and robotic manipulation environments demonstrates the robustness of our mode-based imitation methods under external perturbations.
- **OpenReview**: https://openreview.net/pdf?id=qoHeuRAcSl
        
</details>

### Large Language Models as Tool Makers
> This study presents a novel framework for large language models (LLMs), called LATM, where LLMs create reusable tools for problem-solving. By allowing LLMs to make their own tools, LATM improves the efficiency of subsequent requests, optimizes serving costs by assigning tool-making to more powerful models and tool-using to lightweight models, and offers a functional cache that stores tool functionality instead of LLM responses.

<details>
<summary>Details</summary>
- **Abstract**: Recent research has highlighted the potential of large language models (LLMs) to improve their problem-solving capabilities with the aid of suitable external tools. In our work, we further advance this concept by introducing a closed- loop framework, referred to as LLMs A s Tool Makers (LATM), where LLMs create their own reusable tools for problem-solving. Our approach consists of two phases: 1) tool making: an LLM acts as the tool maker that crafts tools for a set of tasks, where a tool is implemented as a Python utility function. 2) tool using: another LLM acts as the tool user, which applies the tool built by the tool maker for problem-solving. The tool user can be either the same or a different LLM from the tool maker. On the problem-solving server side, tool-making enables continual tool generation and caching as new requests emerge. This framework enables subsequent requests to access cached tools via their corresponding APIs, enhancing the efficiency of task resolution. Beyond enabling LLMs to create their own tools, our framework also uncovers intriguing opportunities to optimize the serving cost of LLMs: Recognizing that tool-making requires more sophisticated capabilities, we assign this task to a powerful, albeit resource-intensive, model. Conversely, the simpler tool-using phase is delegated to a lightweight model. This strategic division of labor allows the once-off cost of tool-making to be spread over multiple instances of tool-using, significantly reducing average costs while maintaining strong performance. Furthermore, our method offers a functional cache through the caching and reuse of tools, which stores the functionality of a class of requests instead of the natural language responses from LLMs, thus extending the applicability of the conventional cache mechanism. We evaluate our approach across various complex reasoning tasks, including Big-Bench tasks. With GPT-4 as the tool maker and GPT-3.5 as the tool user, LATM demonstrates performance equivalent to using GPT-4 for both roles, but with a significantly reduced inference cost.
- **OpenReview**: https://openreview.net/pdf?id=qV83K9d5WB
        
</details>

### Learning Grounded Action Abstractions from Language
> This study presents a novel method for learning useful action abstractions and corresponding low-level policies by utilizing large language models (LLMs) and integrating them into a formal hierarchical planning system. The proposed approach significantly enhances the accuracy and generalization of planning in language-guided interactive planning domains.

<details>
<summary>Details</summary>
- **Abstract**: Long-horizon planning is dauntingly hard -- it requires modeling relevant aspects of the environment and searching over large, complex action spaces. \\textit{Hierarchical planning} approaches make complex problems more tractable using temporal \\textit{action abstractions}, decomposing hard tasks into smaller abstract subproblems that can be solved modularly. However, actually learning useful action abstractions has long posed significant challenges without human expert knowledge. Here, we introduce a system that leverages background information in language to learn a \\textit{library of symbolic action abstractions and accompanying low-level policies} that can be composed to solve increasingly complex tasks. Our approach queries large language models (LLMs) as a prior for proposing useful symbolic action definitions, but integrates these proposals into a formal hierarchical planning system to ground and verify proposed actions. On two language-guided interactive planning domains (\\textit{Mini Minecraft} and the \\textit{ALFRED Household Tasks} benchmark), our approach far outperforms other baseline approaches that use LLMs in planning, enabling far more accurate planning and enable better generalization to more complex tasks.
- **OpenReview**: https://openreview.net/pdf?id=qJ0Cfj4Ex9
        
</details>

### LayoutNUWA: Revealing the Hidden Layout Expertise of Large Language Models
> LayoutNUWA, a model that combines code generation and large language model knowledge, enhances semantic information and offers a highly interpretable and transparent layout generation procedure, outperforming previous methods with state-of-the-art results.

<details>
<summary>Details</summary>
- **Abstract**: Graphic layout generation, a growing research field, plays a significant role in user engagement and information perception.  Existing methods primarily treat layout generation as a numerical optimization task, focusing on quantitative aspects while overlooking the semantic information of layout, such as the relationship between each layout element.  In this paper, we propose LayoutNUWA, the first model that treats layout generation as a code generation task to enhance semantic information and harness the hidden layout expertise of large language models~(LLMs).  Concretely, we develop a Code Instruct Tuning (CIT) approach comprising three interconnected modules: 1) the Code Initialization (CI) module quantifies the numerical conditions and initializes them as HTML code with strategically placed masks; 2) the Code Completion (CC) module employs the formatting knowledge of LLMs to fill in the masked portions within the HTML code; 3) the Code Rendering (CR) module transforms the completed code into the final layout output, ensuring a highly interpretable and transparent layout generation procedure that directly maps code to a visualized layout. We attain significant state-of-the-art performance (even over 50% improvements compared to previous works) on multiple datasets, showcasing the strong capabilities of LayoutNUWA.
- **OpenReview**: https://openreview.net/pdf?id=qCUWVT0Ayy
        
</details>

### Boosting of Thoughts: Trial-and-Error Problem Solving with Large Language Models
> This paper presents Boosting of Thoughts (BoT), a framework that explores and self-evaluates reasoning steps to enhance prompting for complex problem-solving with LLMs. By leveraging trial-and-error experiences and error analysis obtained from the LLM, BoT iteratively refines prompts and improves reasoning step generation, leading to superior problem-solving performance.

<details>
<summary>Details</summary>
- **Abstract**: The reasoning performance of Large Language Models (LLMs) on a wide range of problems critically relies on chain-of-thought prompting, which involves providing a few chain of thought demonstrations as exemplars in prompts. Recent work, e.g., Tree of Thoughts, has pointed out the importance of exploration and self-evaluation in reasoning step selection for complex problem solving. In this paper, we present Boosting of Thoughts (BoT), an automated prompting framework for problem solving with LLMs by iteratively exploring and self-evaluating many trees of thoughts in order to acquire an ensemble of trial-and-error reasoning experiences, which will serve as a new form of prompting to solve the complex problem. Starting from a simple prompt without requiring examples, BoT iteratively explores and evaluates a large collection of reasoning steps, and more importantly, uses error analysis obtained from the LLM on them to explicitly revise prompting, which in turn enhances reasoning step generation, until a final answer is attained. Our experiments with GPT-4 and Llama2 across extensive complex mathematical problems demonstrate that BoT consistently achieves higher or comparable problem-solving rates than other advanced prompting approaches.
- **OpenReview**: https://openreview.net/pdf?id=qBL04XXex6
        
</details>

### DQ-LoRe: Dual Queries with Low Rank Approximation Re-ranking for In-Context Learning
> DQ-LoRe proposes a novel framework using dual queries and low-rank approximation to select exemplars for in-context learning, leading to significant improvements in performance and adaptability for reasoning tasks with large language models.

<details>
<summary>Details</summary>
- **Abstract**: Recent advances in natural language processing, primarily propelled by Large Language Models (LLMs), have showcased their remarkable capabilities grounded in in-context learning. A promising avenue for guiding LLMs in intricate reasoning tasks involves the utilization of intermediate reasoning steps within the Chain-of-Thought (CoT) paradigm. Nevertheless, the central challenge lies in the effective selection of exemplars for facilitating in-context learning. In this study, we introduce a framework that leverages Dual Queries and Low-rank approximation Re-ranking (DQ-LoRe) to automatically select exemplars for in-context learning. Dual Queries first query LLM to obtain LLM-generated knowledge such as CoT, then query the retriever to obtain the final exemplars via both question and the knowledge. Moreover, for the second query, LoRe employs dimensionality reduction techniques to refine exemplar selection, ensuring close alignment with the input question's knowledge. Through extensive experiments, we demonstrate that DQ-LoRe significantly outperforms prior state-of-the-art methods in the automatic selection of exemplars for GPT-4, enhancing performance from 92.5% to 94.2%. Our comprehensive analysis further reveals that DQ-LoRe consistently outperforms retrieval-based approaches in terms of both performance and adaptability, especially in scenarios characterized by distribution shifts. DQ-LoRe pushes the boundaries of in-context learning and opens up new avenues for addressing complex reasoning challenges.
- **OpenReview**: https://openreview.net/pdf?id=qAoxvePSlq
        
</details>

### #InsTag: Instruction Tagging for Analyzing Supervised Fine-tuning of Large Language Models
> InsTag, a method for tagging the semantics and intentions of human instructions for training language models, enables the quantification of instruction diversity and complexity. This discovery led to the development of TagLM, a language model that outperforms existing models by utilizing more diverse and complex instructions in its training data, highlighting the significance of this aspect for accurate instruction-following abilities.

<details>
<summary>Details</summary>
- **Abstract**: Pre-trained large language models (LLMs) can understand and align with human instructions by supervised fine-tuning (SFT). It is commonly believed that diverse and complex SFT data are of the essence to enable good instruction-following abilities. However, such diversity and complexity are obscure and lack quantitative analyses. In this work, we propose InsTag, an open-set instruction tagging method, to identify semantics and intentions of human instructions by tags that provide access to definitions and quantified analyses of instruction diversity and complexity. We obtain 6.6K fine-grained tags to describe instructions from popular open-sourced SFT datasets comprehensively. We find that the abilities of aligned LLMs benefit from more diverse and complex instructions in SFT data. Based on this observation, we propose a data sampling procedure based on InsTag, and select 6K diverse and complex samples from open-source datasets for SFT. The resulting models, TagLM, outperform open-source models based on considerably larger SFT data evaluated by MT-Bench, echoing the importance of instruction diversity and complexity and the effectiveness of InsTag. InsTag has robust potential to be extended to more applications beyond the data selection as it provides an effective way to analyze the distribution of instructions.
- **OpenReview**: https://openreview.net/pdf?id=pszewhybU9
        
</details>

### Jailbreak in pieces: Compositional Adversarial Attacks on Multi-Modal Language Models
> Vision language models, which integrate image and text, are susceptible to cross-modality attacks that compromise their alignment. By combining adversarial images with generic prompts, attackers can exploit vulnerabilities in the alignment mechanism, leveraging the image component to influence the LLM's responses, raising concerns about the security of these models.

<details>
<summary>Details</summary>
- **Abstract**: We introduce new jailbreak attacks on vision language models (VLMs), which use aligned LLMs and are resilient to text-only jailbreak attacks. Specifically, we develop cross-modality attacks on alignment where we pair adversarial images going through the vision encoder with textual prompts to break the alignment of the language model. Our attacks employ a novel compositional strategy that combines an image, adversarially targeted towards toxic embeddings, with generic prompts to accomplish the jailbreak. Thus, the LLM draws the context to answer the generic prompt from the adversarial image. The generation of benign-appearing adversarial images leverages a novel embedding-space-based methodology, operating with no access to the LLM model. Instead, the attacks require access only to the vision encoder and utilize one of our four embedding space targeting strategies. By not requiring access to the LLM, the attacks lower the entry barrier for attackers, particularly when vision encoders such as CLIP are embedded in closed-source LLMs. The attacks achieve a high success rate across different VLMs, highlighting the risk of cross-modality alignment vulnerabilities, and the need for new alignment approaches for multi-modal models.
- **OpenReview**: https://openreview.net/pdf?id=plmBsXHxgR
        
</details>

### SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning
> SelfCheck, a novel zero-shot verification schema, allows LLMs to identify errors in their step-by-step reasoning, leading to improved question-answering accuracy, especially for non-linear problems.

<details>
<summary>Details</summary>
- **Abstract**: The recent progress in large language models (LLMs), especially the invention of chain-of-thought prompting, has made it possible to automatically answer questions by stepwise reasoning. However, when faced with more complicated problems that require non-linear thinking, even the strongest LLMs make mistakes.  To address this, we explore whether LLMs are able to recognize errors in their own step-by-step reasoning, without resorting to external resources. To this end, we propose SelfCheck, a general-purpose zero-shot verification schema for recognizing such errors. We then use the results of these checks to improve question-answering performance by conducting weighted voting on multiple solutions to the question. We test SelfCheck on three datasets---GSM8K, MathQA, and MATH---and find that it successfully recognizes errors and, in turn, increases final answer accuracies.
- **OpenReview**: https://openreview.net/pdf?id=pTHfApDakA
        
</details>

### RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems
> RepoBench tackles the assessment gap in multi-file code auto-completion by introducing three interconnected tasks (retrieval, code completion, and pipeline) to facilitate a more comprehensive evaluation and foster advancements in real-world programming scenarios.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have greatly advanced code auto-completion systems, with a potential for substantial productivity enhancements for developers. However, current benchmarks mainly focus on single-file tasks, leaving an assessment gap for more complex, real-world, multi-file programming scenarios. To fill this gap, we introduce RepoBench, a new benchmark specifically designed for evaluating repository-level code auto-completion systems. RepoBench consists of three interconnected evaluation tasks: RepoBench-R (Retrieval), RepoBench-C (Code Completion), and RepoBench-P (Pipeline). Each task respectively measures the system's ability to retrieve the most relevant code snippets from other files as cross-file context, predict the next line of code with cross-file and in-file context, and handle complex tasks that require a combination of both retrieval and next-line prediction. RepoBench aims to facilitate a more complete comparison of performance and encouraging continuous improvement in auto-completion systems.
- **OpenReview**: https://openreview.net/pdf?id=pPjZIOuQuF
        
</details>

### RAIN: Your Language Models Can Align Themselves without Finetuning
> This research explores a novel inference method (RAIN) that enables unaligned LLMs to align with human preferences without the need for additional data or training. RAIN leverages self-evaluation and rewind mechanisms, allowing LLMs to assess their own generations and modify them to match human expectations, resulting in improved consistency with human preferences on real-world datasets, as demonstrated through evaluations with GPT-4 and human raters.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) often demonstrate inconsistencies with human preferences. Previous research typically gathered human preference data and then aligned the pre-trained models using reinforcement learning or instruction tuning, a.k.a. the finetuning step. In contrast, aligning frozen LLMs without requiring alignment data is more appealing. This work explores the potential of the latter setting. We discover that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting. We introduce a novel inference method, Rewindable Auto-regressive INference (RAIN), that allows pre-trained LLMs to evaluate their own generation and use the evaluation results to guide rewind and generation for AI safety. Notably, RAIN operates without the need of extra data for model alignment and abstains from any training, gradient computation, or parameter updates. Experimental results evaluated by GPT-4 and humans demonstrate the effectiveness of RAIN: on the HH dataset, RAIN improves the harmlessness rate of LLaMA 30B from 82% of vanilla inference to 97%, while maintaining the helpfulness rate. On the TruthfulQA dataset, RAIN improves the truthfulness of the already-well-aligned LLaMA-2-chat 13B model by 5%.
- **OpenReview**: https://openreview.net/pdf?id=pETSfWMUzy
        
</details>

### Two-stage LLM Fine-tuning with Less Specialization and More Generalization
> Fine-tuning large language models on specific tasks typically reduces their general problem-solving abilities due to format specialization. Prompt Tuning with Model Tuning (ProMoT) addresses this issue by separating task-specific format learning into additional parameters, leading to improved generalization on both fine-tuned and out-of-domain tasks.

<details>
<summary>Details</summary>
- **Abstract**: Pretrained large language models (LLMs) are general purpose problem solvers applicable to a diverse set of tasks with prompts. They can be further improved towards a specific task by fine-tuning on a specialized dataset. However, fine-tuning usually makes the model narrowly specialized on this dataset with reduced general in-context learning performances, which is undesirable whenever the fine-tuned model needs to handle additional tasks where no fine-tuning data is available.  In this work, we first demonstrate that fine-tuning on a single task indeed decreases LLMs' general in-context learning performance. We discover one important cause of such forgetting, format specialization, where the model overfits to the format of the fine-tuned task. We further show that format specialization happens at the very beginning of fine-tuning. To solve this problem, we propose Prompt Tuning with MOdel Tuning (ProMoT), a simple yet effective two-stage fine-tuning framework that reduces format specialization and improves generalization. ProMoT offloads task-specific format learning into additional and removable parameters by first doing prompt tuning and then fine-tuning the model itself with this soft prompt attached.  With experiments on several fine-tuning tasks and 8 in-context evaluation tasks, we show that ProMoT achieves comparable performance on fine-tuned tasks to standard fine-tuning, but with much less loss of in-context learning performances across a board range of  out-of-domain evaluation tasks. More importantly, ProMoT can even enhance generalization on in-context learning tasks that are semantically related to the fine-tuned task, e.g. ProMoT on En-Fr translation significantly improves performance on other language pairs, and ProMoT on NLI improves performance on summarization. Experiments also show that ProMoT can improve the generalization performance of  multi-task training.
- **OpenReview**: https://openreview.net/pdf?id=pCEgna6Qco
        
</details>

### Causal Modelling Agents: Causal Graph Discovery through Synergising Metadata- and Data-driven Reasoning
> The paper presents the Causal Modelling Agent (CMA), a framework that combines the reasoning capabilities of Large Language Models (LLMs) with the data-driven modelling of Deep Structural Causal Models (DSCMs) for causal discovery. The CMA outperforms previous data-driven and metadata-driven approaches in benchmarks and provides new insights into the causal relationships among biomarkers of Alzheimer's Disease (AD).

<details>
<summary>Details</summary>
- **Abstract**: Scientific discovery hinges on the effective integration of metadata, which refers to a set of 'cognitive' operations such as determining what information is relevant for inquiry, and data, which encompasses physical operations such as observation and experimentation. This paper introduces the Causal Modelling Agent (CMA), a novel framework that synergizes the metadata-based reasoning capabilities of Large Language Models (LLMs) with the data-driven modelling of Deep Structural Causal Models (DSCMs) for the task of causal discovery. We evaluate the CMA's performance on a number of benchmarks, as well as on the real-world task of modelling the clinical and radiological phenotype of Alzheimer's Disease (AD). Our experimental results indicate that the CMA can outperform previous data-driven or metadata-driven approaches to causal discovery. In our real-world application, we use the CMA to derive new insights into the causal relationships among biomarkers of AD.
- **OpenReview**: https://openreview.net/pdf?id=pAoqRlTBtY
        
</details>

### The Truth Is In There: Improving Reasoning with Layer-Selective Rank Reduction
> Transformer-based Large Language Models (LLMs) can be significantly improved by removing higher-order components from their multi-layer perception layers, leading to dramatic performance boosts in question-answering tasks and various modalities.

<details>
<summary>Details</summary>
- **Abstract**: Transformer-based Large Language Models (LLMs) have become a fixture in modern machine learning. Correspondingly, significant resources are allocated towards research that aims to further advance this technology, typically resulting in models of increasing size that are trained on increasing amounts of data. This work, however, demonstrates the surprising result that it is often possible to im- prove the performance of LLMs by simply removing higher-order components of their constituent weight matrices in the multi-layer perception (MLP) layers. This simple intervention, which we call LAyer-SElective Rank reduction (LASER), can be done on a model after training has completed, and requires no additional parameters or data. LASER can dramatically boost predictive performanceat times by 80% over the models original performanceon question-answering tasks and across various modalities for which Transformers are used.
- **OpenReview**: https://openreview.net/pdf?id=ozX92bu8VA
        
</details>

### ReLU Strikes Back: Exploiting Activation Sparsity in Large Language Models
> This study suggests using the computationally efficient ReLU activation function in Large Language Models (LLMs) to drastically reduce computation during inference without significantly compromising performance. The ReLU activation also enables the reutilization of activated neurons, leading to further computational reduction strategies, making LLMs more feasible for resource-constrained devices.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) with billions of parameters have drastically transformed AI applications. However, their demanding computation during inference has raised significant challenges for deployment on resource-constrained devices. Despite recent trends favoring alternative activation functions such as GELU or SiLU, known for increased computation, this study strongly advocates for reinstating ReLU activation in LLMs. We demonstrate that using the ReLU activation function has a negligible impact on convergence and performance while significantly reducing computation and weight transfer. This reduction is particularly valuable during the memory-bound inference step, where efficiency is paramount. Exploring sparsity patterns in ReLU-based LLMs, we unveil the reutilization of activated neurons for generating new tokens and leveraging these insights, we propose practical strategies to substantially reduce LLM inference computation up to three times, using ReLU activations with minimal performance trade-offs.
- **OpenReview**: https://openreview.net/pdf?id=osoWxY8q2E
        
</details>

### AffineQuant: Affine Transformation Quantization for Large Language Models
> AffineQuant significantly improves Post-Training Quantization (PTQ) for Large-scale Language Models (LLMs) by extending the optimization scope to equivalent Affine transformations, resulting in reduced quantization errors, especially in low-bit configurations, enabling efficient and accurate deployment of large models on edge devices.

<details>
<summary>Details</summary>
- **Abstract**: The significant resource requirements associated with Large-scale Language Models (LLMs) have generated considerable interest in the development of techniques aimed at compressing and accelerating neural networks.  Among these techniques, Post-Training Quantization (PTQ) has emerged as a subject of considerable interest due to its noteworthy compression efficiency and cost-effectiveness in the context of training. Existing PTQ methods for LLMs limit the optimization scope to scaling transformations between pre- and post-quantization weights.  This constraint results in significant errors after quantization, particularly in low-bit configurations.  In this paper, we advocate for the direct optimization using equivalent Affine transformations in PTQ (AffineQuant).  This approach extends the optimization scope and thus significantly minimizing quantization errors.  Additionally, by employing the corresponding inverse matrix, we can ensure equivalence between the pre- and post-quantization outputs of PTQ, thereby maintaining its efficiency and generalization capabilities.  To ensure the invertibility of the transformation during optimization, we further introduce a gradual mask optimization method.  This method initially focuses on optimizing the diagonal elements and gradually extends to the other elements.  Such an approach aligns with the Levy-Desplanques theorem, theoretically ensuring invertibility of the transformation.  As a result, significant performance improvements are evident across different LLMs on diverse datasets.  Notably, these improvements are most pronounced when using very low-bit quantization, enabling the deployment of large models on edge devices.  To illustrate, we attain a C4 perplexity of $14.89$ ({ 10.00$\\downarrow$} vs $24.89$ in OmniQuant) on the LLaMA-$7$B model of W$2$A$16$ quantization. AffineQuant significantly outperforms OmniQuant on smaller models, achieving a perplexity of $42.29$ ({ 33.14$\\downarrow$} vs $75.43$ in OmniQuant) when using $2$-bit $128$-group quantization for OPT-$125$M, which setting a new state-of-the-art benchmark for PTQ in LLMs.  Codes are available in the supplementary materials.
- **OpenReview**: https://openreview.net/pdf?id=of2rhALq8l
        
</details>

### Privacy-Preserving In-Context Learning with Differentially Private Few-Shot Generation
> By safeguarding sensitive information while enabling effective in-context learning with LLMs, the study advocates for the use of synthetic few-shot demonstrations generated with differential privacy, thus broadening the scope of privacy-protected ICL applications.

<details>
<summary>Details</summary>
- **Abstract**: We study the problem of in-context learning (ICL) with large language models (LLMs) on private datasets.  This scenario poses privacy risks, as LLMs may leak or regurgitate the private examples demonstrated in the prompt. We propose a novel algorithm that generates synthetic few-shot demonstrations from the private dataset with formal differential privacy (DP) guarantees, and show empirically that it can achieve effective ICL. We conduct extensive experiments on standard benchmarks and compare our algorithm with non-private ICL and zero-shot solutions.  Our results demonstrate that our algorithm can achieve competitive performance with strong privacy levels. These results open up new possibilities for ICL with privacy protection for a broad range of applications.
- **OpenReview**: https://openreview.net/pdf?id=oZtt0pRnOl
        
</details>

### ExeDec: Execution Decomposition for Compositional Generalization in Neural Program Synthesis
> Can neural program synthesis techniques break down complex tasks into smaller ones like humans do? ExeDec, a novel strategy, shows promising results in tackling this challenge, improving the ability of models to solve complex tasks by systematically decomposing them.

<details>
<summary>Details</summary>
- **Abstract**: When writing programs, people have the ability to tackle a new complex task by decomposing it into smaller and more familiar subtasks. While it is difficult to measure whether neural program synthesis methods have similar capabilities, we can measure whether they compositionally generalize, that is, whether a model that has been trained on the simpler subtasks is subsequently able to solve more complex tasks. In this paper, we characterize several different forms of compositional generalization that are desirable in program synthesis, forming a meta-benchmark which we use to create generalization tasks for two popular datasets, RobustFill and DeepCoder. We then propose ExeDec, a novel decomposition-based synthesis strategy that predicts execution subgoals to solve problems step-by-step informed by program execution at each step. When used with Transformer models trained from scratch, ExeDec has better synthesis performance and greatly improved compositional generalization ability compared to baselines. Finally, we use our benchmarks to demonstrate that LLMs struggle to compositionally generalize when asked to do programming-by-example in a few-shot setting, but an ExeDec-style prompting approach can improve the generalization ability and overall performance.
- **OpenReview**: https://openreview.net/pdf?id=oTRwljRgiv
        
</details>

### Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph
> This paper presents "Think-on-Graph" (ToG), a novel approach that integrates large language models (LLMs) with external knowledge graphs (KGs). ToG empowers LLMs with enhanced reasoning abilities by enabling them to iteratively explore KGs and retrieve knowledge relevant to the task at hand.

<details>
<summary>Details</summary>
- **Abstract**: Although large language models (LLMs) have achieved significant success in various tasks, they often struggle with hallucination problems, especially in scenarios requiring deep and responsible reasoning. These issues could be partially addressed by introducing external knowledge graphs (KG) in LLM reasoning. In this paper, we propose a new LLM-KG integrating paradigm ``$\\hbox{LLM}\\otimes\\hbox{KG}$'' which treats the LLM as an agent to interactively explore related entities and relations on KGs and perform reasoning based on the retrieved knowledge. We further implement this paradigm by introducing a new approach called Think-on-Graph (ToG), in which the LLM agent iteratively executes beam search on KG, discovers the most promising reasoning paths, and returns the most likely reasoning results. We use a number of well-designed experiments to examine and illustrate the following advantages of ToG: 1) compared with LLMs, ToG has better deep reasoning power; 2) ToG has the ability of knowledge traceability and knowledge correctability by leveraging LLMs reasoning and expert feedback; 3) ToG provides a flexible plug-and-play framework for different LLMs, KGs and prompting strategies without any additional training cost; 4) the performance of ToG with small LLM models could exceed large LLM such as GPT-4 in certain scenarios and this reduces the cost of LLM deployment and application. As a training-free method with lower computational cost and better generality, ToG achieves overall SOTA in 6 out of 9 datasets where most previous SOTAs rely on additional training.
- **OpenReview**: https://openreview.net/pdf?id=nnVO1PvbTv
        
</details>

### Listen, Think, and Understand
> LTU emerges as a multimodal model combining audio perception and reasoning abilities. Trained on a diverse dataset of audio, questions, and answers, it not only classifies and captions audio signals but also infers, thinks, and comprehends audio content, surpassing existing audio models.

<details>
<summary>Details</summary>
- **Abstract**: The ability of artificial intelligence (AI) systems to perceive and comprehend audio signals is crucial for many applications. Although significant progress has been made in this area since the development of AudioSet, most existing models are designed to map audio inputs to pre-defined, discrete sound label sets. In contrast, humans possess the ability to not only classify sounds into general categories, but also to listen to the finer details of the sounds, explain the reason for the predictions, think about what the sound infers, and understand the scene and what action needs to be taken, if any. Such capabilities beyond perception are not yet present in existing audio models. On the other hand, modern large language models (LLMs) exhibit emerging reasoning ability but they lack audio perception capabilities. Therefore, we ask the question: can we build a model that has both audio perception and a reasoning ability?   In this paper, we propose a new audio foundation model, called LTU (Listen, Think, and Understand). To train LTU, we created a new OpenAQA-5M dataset consisting of 1.9 million closed-ended and 3.7 million open-ended, diverse (audio, question, answer) tuples, and have used an autoregressive training framework with a perception-to-understanding curriculum. LTU demonstrates strong performance and generalization ability on conventional audio tasks such as classification and captioning. More importantly, it exhibits emerging audio reasoning and comprehension abilities that are absent in existing audio models. To the best of our knowledge, LTU is one of the first multimodal large language models that focus on general audio (rather than just speech) understanding.
- **OpenReview**: https://openreview.net/pdf?id=nBZBPXdJlC
        
</details>

### HAZARD Challenge: Embodied Decision Making in Dynamically Changing Environments
> The paper provides a new simulated benchmark, called HAZARD, to evaluate embodied agents' decision-making abilities in dynamic environments, utilizing large language models for common sense reasoning and exploring the promise and challenges of this approach.

<details>
<summary>Details</summary>
- **Abstract**: Recent advances in high-fidelity virtual environments serve as one of the major driving forces for building intelligent embodied agents to perceive, reason and interact with the physical world. Typically, these environments remain unchanged unless agents interact with them. However, in real-world scenarios, agents might also face dynamically changing environments characterized by unexpected events and need to rapidly take action accordingly. To remedy this gap, we propose a new simulated embodied benchmark, called HAZARD, specifically designed to assess the decision-making abilities of embodied agents in dynamic situations. HAZARD consists of three unexpected disaster scenarios, including fire, flood, and wind, and specifically supports the utilization of large language models (LLMs) to assist common sense reasoning and decision-making. This benchmark enables us to evaluate autonomous agents' decision-making capabilities across various pipelines, including reinforcement learning (RL), rule-based, and search-based methods in dynamically changing environments. As a first step toward addressing this challenge using large language models, we further develop an LLM-based agent and perform an in-depth analysis of its promise and challenge of solving these challenging tasks.
- **OpenReview**: https://openreview.net/pdf?id=n6mLhaBahJ
        
</details>

### OctoPack: Instruction Tuning Code Large Language Models
> Instruction tuning of large language models using CommitPack, a large dataset of Git commits, achieves state-of-the-art performance on natural coding tasks, demonstrating the effectiveness of code instructions in enhancing model capabilities.

<details>
<summary>Details</summary>
- **Abstract**: Finetuning large language models (LLMs) on instructions leads to vast performance improvements on natural language tasks. We apply instruction tuning using code, leveraging the natural structure of Git commits, which pair code changes with human instructions.  We compile CommitPack: 4 terabytes of Git commits across 350 programming languages. We benchmark CommitPack against other natural and synthetic code instructions (xP3x, Self-Instruct, OASST) on the 16B parameter StarCoder model, and achieve state-of-the-art performance among models not trained on OpenAI outputs, on the HumanEval Python benchmark (46.2% pass@1). We further introduce HumanEvalPack, expanding the HumanEval benchmark to a total of 3 coding tasks (Code Repair, Code Explanation, Code Synthesis) across 6 languages (Python, JavaScript, Java, Go, C++, Rust). Our models, OctoCoder and OctoGeeX, achieve the best performance across HumanEvalPack among all permissive models, demonstrating CommitPack's benefits in generalizing to a wider set of languages and natural coding tasks. Code, models and data will be made freely available.
- **OpenReview**: https://openreview.net/pdf?id=mw1PWNSWZP
        
</details>

### Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding
> Motivated by human thinking, this research introduces SoT, a data-centric approach that reduces LLM generation latency by guiding the models to create an answer skeleton and filling it in parallel, yielding speed-ups and potentially better response quality.

<details>
<summary>Details</summary>
- **Abstract**: This work aims at decreasing the end-to-end generation latency of large language models (LLMs). One of the major causes of the high generation latency is the sequential decoding approach adopted by almost all state-of-the-art LLMs. In this work, motivated by the thinking and writing process of humans, we propose Skeleton-of-Thought (SoT), which first guides LLMs to generate the skeleton of the answer, and then conducts parallel API calls or batched decoding to complete the contents of each skeleton point in parallel. Not only does SoT provide considerable speed-ups across 12 LLMs, but it can also potentially improve the answer quality on several question categories. SoT is an initial attempt at data-centric optimization for inference efficiency, and further underscores the potential of pushing LLMs to think more like a human for answer quality.
- **OpenReview**: https://openreview.net/pdf?id=mqVgBbNCm9
        
</details>

### Improving Code Style for Accurate Code Generation
> By enhancing code structure, readability, and incorporating planning annotations, a novel data-cleaning pipeline significantly improves code generation performance by up to 30%, even outperforming larger closed-source models.

<details>
<summary>Details</summary>
- **Abstract**: Natural language to code generation is an important application area of LLMs and has received wide attention from the community.  The majority of relevant studies have exclusively concentrated on increasing the quantity and functional correctness of training sets while disregarding other stylistic elements of programs. More recently, data quality has garnered a lot of interest and multiple works have showcased its importance for improving performance. In this work, we investigate data quality for code and find that making the code more structured and readable leads to improved code generation performance of the system. We build a novel data-cleaning pipeline that uses these principles to transform existing programs by 1.) renaming variables, 2.) modularizing and decomposing complex code into smaller helper sub-functions, and 3.) inserting natural-language based planning annotations. We evaluate our approach on two challenging algorithmic code generation benchmarks and find that fine-tuning CodeLLaMa-7B on our transformed programs improves the performance by up to \\textbf{30%} compared to fine-tuning on the original dataset. Additionally, we demonstrate improved performance from using a smaller amount of higher-quality data, finding that a model fine-tuned on the entire original dataset is outperformed by a model trained on one-eighth of our cleaned dataset. Even in comparison to closed-source models, our models outperform the much larger AlphaCode models.
- **OpenReview**: https://openreview.net/pdf?id=maRYffiUpI
        
</details>

### LLM Blueprint: Enabling Text-to-Image Generation with Complex and Detailed Prompts
> By breaking down complex text prompts into key components and employing an iterative refinement process, the presented approach significantly improves the fidelity of diffusion-based text-to-image generation models when handling lengthy and intricate descriptions, allowing for the creation of more coherent and detailed scenes.

<details>
<summary>Details</summary>
- **Abstract**: Diffusion-based generative models have significantly advanced text-to-image generation but encounter challenges when processing lengthy and intricate text prompts describing complex scenes with multiple objects. While excelling in generating images from short, single-object descriptions, these models often struggle to faithfully capture all the nuanced details within longer and more elaborate textual inputs. In response, we present a novel approach leveraging Large Language Models (LLMs) to extract critical components from text prompts, including bounding box coordinates for foreground objects, detailed textual descriptions for individual objects, and a succinct background context. These components form the foundation of our layout-to-image generation model, which operates in two phases. The initial Global Scene Generation utilizes object layouts and background context to create an initial scene but often falls short in faithfully representing object characteristics as specified in the prompts. To address this limitation, we introduce an Iterative Refinement Scheme that iteratively evaluates and refines box-level content to align them with their textual descriptions, recomposing objects as needed to ensure consistency. Our evaluation on complex prompts featuring multiple objects demonstrates a substantial improvement in recall compared to baseline diffusion models. This is further validated by a user study, underscoring the efficacy of our approach in generating coherent and detailed scenes from intricate textual inputs. Our iterative framework offers a promising solution for enhancing text-to-image generation models' fidelity with lengthy, multifaceted descriptions, opening new possibilities for accurate and diverse image synthesis from textual inputs.
- **OpenReview**: https://openreview.net/pdf?id=mNYF0IHbRy
        
</details>

### Beyond task performance: evaluating and reducing the flaws of large multimodal models with in-context-learning
> This paper evaluates Large Multimodal Models (LMMs) on five axes and discovers significant flaws, despite their success. In-Context Learning (ICL) is explored to address these limitations, leading to the development of new multimodal ICL approaches that effectively tackle certain flaws, such as improved explainability and instruction following.

<details>
<summary>Details</summary>
- **Abstract**: Following the success of Large Language Models (LLMs), Large Multimodal Models (LMMs), such as the Flamingo model and its subsequent competitors, have started to emerge as natural step towards generalist agents. However, interacting with recent LMMs reveals major limitations that are hardly captured by the current evaluation benchmarks. Indeed, task performances (e.g., VQA accuracy) alone do not provide enough clues to understand their real capabilities, limitations, and to which extent such models are aligned to human expectations. To refine our understanding on those flaws, we deviate from the current evaluation paradigm, and (1) evaluate 8 recent open-source LMMs (based on the Flamingo architecture such as OpenFlamingo and IDEFICS) on 5 different axes; hallucinations, abstention, compositionality, explainability and instruction following. Our evaluation on these axes reveals major flaws in LMMs. To efficiently address these problems, and inspired by the success of In-Context Learning (ICL) in LLMs, (2) we explore ICL as a solution, and study how it affects these limitations. Based on our ICL study, (3) we push ICL further, and propose new multimodal ICL approaches such as; Multitask-ICL, Chain-of-Hindsight-ICL and Self-Correcting-ICL. Our findings are as follows. (1) Despite their success, LMMs have flaws that remain unsolved with scaling alone. (2) The effect of ICL on LMMs flaws is nuanced; despite its effectiveness for improved explainability, abstention and instruction following, ICL does not improve compositional abilities, and actually even amplifies hallucinations. (3) The proposed ICL variants are promising as post-hoc approaches to efficiently tackle some of those flaws. The code will be made public.
- **OpenReview**: https://openreview.net/pdf?id=mMaQvkMzDi
        
</details>

### Seeking Neural Nuggets: Knowledge Transfer in Large Language Models from a Parametric Perspective
> A study explores the transferability of knowledge-specific parameters between Large Language Models (LLMs) of varying sizes, using sensitivity-based extraction and the LoRA module for injection. The results suggest that model parameters can be effectively transferred, emphasizing the importance of parameters in knowledge transfer.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) inherently encode a wealth of knowledge within their parameters through pre-training on extensive corpora. While prior research has delved into operations on these parameters to manipulate the underlying implicit knowledgeencompassing detection, editing, and mergingthere remains an ambiguous understanding regarding their transferability across models with varying scales. In this paper, we seek to empirically investigate knowledge transfer from larger to smaller models through a parametric perspective. To achieve this, we employ sensitivity-based techniques to extract and align knowledge-specific parameters between different LLMs. Moreover, the LoRA module is used as the intermediary mechanism for injecting the extracted knowledge into smaller models. Evaluations across four benchmarks validate the efficacy of our proposed method. Our findings highlight the critical factors contributing to the process of parametric knowledge transfer, underscoring the transferability of model parameters across LLMs of different scales.
- **OpenReview**: https://openreview.net/pdf?id=mIEHIcHGOo
        
</details>

### DENEVIL: TOWARDS DECIPHERING AND NAVIGATING THE ETHICAL VALUES OF LARGE LANGUAGE MODELS VIA INSTRUCTION LEARNING
> This paper explores the ethical values of Large Language Models, discovering their potential for misalignment through a new prompt generation algorithm and dataset. It proposes an in-context alignment method to improve value compliance, highlighting the need for ongoing ethical considerations in integrating LLMs into society.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have made unprecedented breakthroughs, yet their increasing integration into everyday life might raise societal risks due to generated unethical content. Despite extensive study on specific issues like bias, the intrinsic values of LLMs remain largely unexplored from a moral philosophy perspective. This work delves into ethical values utilizing Moral Foundation Theory. Moving beyond conventional discriminative evaluations with poor reliability, we propose DeNEVIL, a novel prompt generation algorithm tailored to dynamically exploit LLMs value vulnerabilities and elicit the violation of ethics in a generative manner, revealing their underlying value inclinations. On such a basis, we construct MoralPrompt, a high-quality dataset comprising 2,397 prompts covering 500+ value principles, and then benchmark the intrinsic values across a spectrum of LLMs. We discovered that most models are essentially misaligned, necessitating further ethical value alignment. In response, we develop VILMO, an in-context alignment method that substantially enhances the value compliance of LLM outputs by learning to generate appropriate value instructions, outperforming existing competitors. Our methods are suitable for black-box and open-source models, offering a promising initial step in studying the ethical values of LLMs.
- **OpenReview**: https://openreview.net/pdf?id=m3RRWWFaVe
        
</details>

### To the Cutoff... and Beyond? A Longitudinal Perspective on LLM Data Contamination
> This study unveils the extent of data contamination in language models (LLMs) by analyzing their performance on benchmarks released over time. The findings suggest that LLMs may exploit examples present in their training data, highlighting the need for careful benchmark creation and evaluation practices to ensure accurate assessment of LLM capabilities.

<details>
<summary>Details</summary>
- **Abstract**: Recent claims about the impressive abilities of large language models (LLMs) are often supported by evaluating publicly available benchmarks.  Since LLMs train on wide swaths of the internet, this practice raises concerns of data contamination, i.e., evaluating on examples that are explicitly or implicitly included in the training data.  Data contamination remains notoriously challenging to measure and mitigate, even with partial attempts like controlled experimentation of training data, canary strings, or embedding similarities.  In this work, we conduct the first thorough longitudinal analysis of data contamination in LLMs by using the natural experiment of training cutoffs in GPT models to look at benchmarks released over time. Specifically, we consider two code/mathematical problem-solving datasets, Codeforces and Project Euler, and find statistically significant trends among LLM pass rate vs. GitHub popularity and release date that provide strong evidence of contamination.  By open-sourcing our dataset, raw results, and evaluation framework, our work paves the way for rigorous analyses of data contamination in modern models. We conclude with a discussion of best practices and future steps for publicly releasing benchmark in the age of LLMs which  train on webscale data.
- **OpenReview**: https://openreview.net/pdf?id=m2NVG4Htxs
        
</details>

### The Cost of Scaling Down Large Language Models: Reducing Model Size Affects Memory before In-context Learning
> Model down-scaling techniques differentially impact large language models' capabilities, with weight pruning significantly impairing factual recall but preserving context processing, while model shrinkage affects recall more than context processing.

<details>
<summary>Details</summary>
- **Abstract**: We study how down-scaling large language model (LLM) size impacts LLM capabilities. We begin by measuring the effects of weight pruning  a popular technique for reducing model size  on the two abilities of LLMs: (a) recalling facts presented during pre-training and (b) processing information presented in context. Surprisingly, we find that existing pruning techniques affect these two abilities of LLMs differently. For example, pruning more than 30% of weights significantly decreases an LLMs ability to recall facts presented during pre-training. Yet pruning 60-70% of weights largely preserves an LLMs ability to process information in-context, ranging from retrieving answers based on information presented in context to learning parameterized functions such as a linear classifier based on a few examples. Moderate pruning impairs LLMs ability to recall facts learnt from pre-training. However, its effect on models ability to process information presented in context is much less pronounced. The said disparate effects similarly arise when replacing the original model with a smaller dense one with reduced width and depth. This similarity suggests that model size reduction in general underpins the said disparity.
- **OpenReview**: https://openreview.net/pdf?id=ldJXXxPE0L
        
</details>

### Grounding Multimodal Large Language Models to the World
> Kosmos-2 is a large language model that can interpret both text and images, allowing it to connect concepts in text with specific things in the physical world. This capability, known as grounding, enhances the model's applications in areas such as visual question answering and referring expression generation, while still maintaining its original abilities in language comprehension and generation.

<details>
<summary>Details</summary>
- **Abstract**: We introduce Kosmos-2, a Multimodal Large Language Model (MLLM), enabling new capabilities of perceiving object descriptions (e.g., bounding boxes) and grounding text to the visual world. Specifically, we represent text spans (i.e., referring expressions and noun phrases) as links in Markdown, i.e., [text span](bounding boxes), where object descriptions are sequences of location tokens. To train the model, we construct a large-scale dataset about grounded image-text pairs (GrIT) together with multimodal corpora. In addition to the existing capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning), Kosmos-2 integrates the grounding capability to downstream applications, while maintaining the conventional capabilities of MLLMs (e.g., perceiving general modalities, following instructions, and performing in-context learning). Kosmos-2 is evaluated on a wide range of tasks, including (i) multimodal grounding, such as referring expression comprehension and phrase grounding, (ii) multimodal referring, such as referring expression generation, (iii) perception-language tasks, and (iv) language understanding and generation. This study sheds a light on the big convergence of language, multimodal perception, and world modeling, which is a key step toward artificial general intelligence. Code can be found in the supplementary material.
- **OpenReview**: https://openreview.net/pdf?id=lLmqxkfSIw
        
</details>

### Bridging Vision and Language Spaces with Assignment Prediction
> VLAP, a unique approach, allows large language models (LLMs) to visually and linguistically analyze information, bridging the gap between language understanding and non-linguistic comprehension. This novel method surpasses previous approaches, enhancing vision-language tasks, and demonstrating LLMs' ability to semantically interpret visual information, opening possibilities for visual semantic reasoning.

<details>
<summary>Details</summary>
- **Abstract**: While pretrained large language models (LLMs) excel in understanding linguistic contexts, it is still an open question: Can LLMs extend their capabilities beyond linguistic contexts to non-linguistic information? This paper introduces VLAP, a novel approach that bridges vision encoders and language models through assignment prediction. Since the LLMs interpret and reason linguistic information from correlations between word embeddings, we harness the well-established word embeddings to map visual representations into language space. Specifically, we simultaneously assign the visual and text representations to a set of word embeddings within LLMs. We propose a new training objective, optimal transport-based assignment prediction, to enforce the consistency of word assignments for paired multimodal data. This allows frozen LLMs to ground their word embedding space in visual data and use their robust semantic taxonomy visually. Moreover, VLAP is memory- and parameter-efficient in that it trains only a single linear layer, and works without extra embedding space (e.g. learnable prototypes) for the assignment prediction. Experimental results show that VLAP achieves substantial improvements over the previous linear transformation-based methods across a range of vision-language tasks, including image captioning, visual question answering, and cross-modal retrieval. We also demonstrate the learned visual representations hold a semantic taxonomy of LLMs, making visual semantic arithmetic possible.
- **OpenReview**: https://openreview.net/pdf?id=lK2V2E2MNv
        
</details>

### Unlock Predictable Scaling from Emergent Abilities
> This study presents a novel evaluation technique, \\textsc{PassUntil}, that unveils the existence of a strict scaling law for task performance and provides insights into the nature of emergent abilities in large language models.

<details>
<summary>Details</summary>
- **Abstract**: The scientific scale-up of large language models (LLMs) necessitates a comprehensive understanding of their scaling properties. However, the existing literature on the scaling properties only yields an incomplete answer: optimization loss decreases predictably as the model size increases, in line with established scaling law; yet no scaling law for task has been established and the task performances are far from predictable during scaling. Task performances typically show minor gains on small models until they improve dramatically once models exceed a size threshold, exemplifying the \\`\\`emergent abilities''. In this study, we discover that small models, although they exhibit minor performance, demonstrate critical and consistent task performance improvements that are not captured by conventional evaluation strategies due to insufficient measurement resolution. To measure such improvements, we introduce \\textsc{PassUntil}, an evaluation strategy with theoretically infinite resolution, through massive sampling in the decoding phase. With \\textsc{PassUntil}, we conduct a quantitative investigation into the scaling law of task performance. The investigation contains two parts. Firstly, a strict \\textsl{task scaling law} that is not conventionally known to exist, is identified, enhancing the predictability of task performances. Remarkably, we are able to predict the performance of the 2.4B model on code generation with merely 0.05% deviation before training starts, which is the first systematic attempt to verify predictable scaling proposed by GPT-4's report. Secondly, underpinned by \\textsc{PassUntil}, we observe concrete evidence of emergent abilities and ascertain that they are not in conflict with the continuity of performance improvement. Their semblance to break-through is that their scaling curve cannot be fitted by standard scaling law function. We then introduce a mathematical definition for the emergent abilities. Through the definition, we refute a prevalent ``multi-step reasoning hypothesis'' regarding the genesis of emergent abilities and propose a new hypothesis with a satisfying fit to the observed scaling curve.
- **OpenReview**: https://openreview.net/pdf?id=lDbjooxLkD
        
</details>

### Successor Heads: Recurring, Interpretable Attention Heads In The Wild
> In this paper, researchers propose "successor heads," a new type of attention head in large language models (LLMs), which can incrementally manipulate tokens with natural ordering, such as numbers or days. The researchers\' analysis suggests that successor heads implement abstract representations common across different LLM architectures and form as early as 31 million model parameters.

<details>
<summary>Details</summary>
- **Abstract**: In this work we present successor heads: attention heads that increment tokens with a natural ordering, such as numbers, months, and days. For example, successor heads increment Monday into Tuesday. We explain the successor head behavior with an approach rooted in mechanistic interpretability, the field that aims to explain how models complete tasks in human-understandable terms. Existing research in this area has found interpretable language model components in small toy models. However, results in toy models have not yet led to insights that explain the internals of frontier models and little is currently understood about the internal operations of large language models. In this paper, we analyze the behavior of successor heads in large language models (LLMs) and find that they implement abstract representations that are common to different architectures. They form in LLMs with as few as 31 million parameters, and at least as many as 12 billion parameters, such as GPT-2, Pythia, and Llama-2. We find a set of mod 10 features that underlie how successor heads increment in LLMs across different architectures and sizes. We perform vector arithmetic with these features to edit head behavior and provide insights into numeric representations within LLMs. Additionally, we study the behavior of successor heads on natural language data, identifying interpretable polysemanticity in a Pythia successor head.
- **OpenReview**: https://openreview.net/pdf?id=kvcbV8KQsi
        
</details>

### Beyond Memorization: Violating Privacy via Inference with Large Language Models
> Pre-trained large language models (LLMs) can infer personal attributes (e.g., location, income, sex) from text with high accuracy and efficiency, raising concerns about privacy violations in LLM applications and the need for enhanced privacy protection measures.

<details>
<summary>Details</summary>
- **Abstract**: Current privacy research on large language models (LLMs) primarily focuses on the issue of extracting memorized training data. At the same time, models inference capabilities have increased drastically. This raises the key question of whether current LLMs could violate individuals privacy by inferring personal attributes from text given at inference time. In this work, we present the first comprehensive study on the capabilities of pretrained LLMs to infer personal attributes from text. We construct a dataset consisting of real Reddit profiles, and show that current LLMs can infer a wide range of personal attributes (e.g., location, income, sex), achieving up to 85% top-1 and 95.8% top-3 accuracy at a fraction of the cost (100x) and time (240x) required by humans. As people increasingly interact with LLM-powered chatbots across all aspects of life, we also explore the emerging threat of privacy-invasive chatbots trying to extract personal information through seemingly benign questions. Finally, we show that common mitigations, i.e., text anonymization and model alignment, are currently ineffective at protecting user privacy against LLM inference. Our findings highlight that current LLMs can infer personal data at a previously unattainable scale. In the absence of working defenses, we advocate for a broader discussion around LLM privacy implications beyond memorization, striving for stronger and wider privacy protection.
- **OpenReview**: https://openreview.net/pdf?id=kmn0BhQk7p
        
</details>

### Large Language Models as Automated Aligners for  benchmarking  Vision-Language Models
> Auto-Bench leverages Large Language Models (LLMs) to explore the alignment of Vision-Language Models (VLMs) with human intelligence, enabling automated data curation and assessment that addresses limitations of existing benchmarks.

<details>
<summary>Details</summary>
- **Abstract**: With the advancements in Large Language Models (LLMs), Vision-Language Models (VLMs) have reached a new level of sophistication, showing notable competence in executing intricate cognition and reasoning tasks. However, existing evaluation benchmarks, primarily relying on rigid, hand-crafted datasets to measure task-specific performance, face significant limitations in assessing the alignment of these increasingly anthropomorphic models with human intelligence. In this work, we address the limitations via Auto-Bench, which delves into exploring LLMs as proficient aligners, measuring the alignment between VLMs and human intelligence and value through automatic data curation and assessment. Specifically, for data curation, Auto-Bench utilizes LLMs (e.g., GPT-4) to automatically generate a vast set of question-answer-reasoning triplets via prompting on visual symbolic representations (e.g., captions, object locations, instance relationships, and etc. The curated data closely matches human intent, owing to the extensive world knowledge embedded in LLMs. Through this pipeline, a total of 28.5K human-verified and 3,504K unfiltered question-answer-reasoning triplets have been curated, covering 4 primary abilities and 16 sub-abilities. We subsequently engage LLMs like GPT-3.5 to serve as judges, implementing the quantitative and qualitative automated assessments to facilitate a comprehensive evaluation of VLMs. Our validation results reveal that LLMs are proficient in both evaluation data curation and model assessment, achieving an average agreement rate of 85%. We envision Auto-Bench as a flexible, scalable, and comprehensive benchmark for evaluating the evolving sophisticated VLMs.
- **OpenReview**: https://openreview.net/pdf?id=kZEXgtMNNo
        
</details>

### Bias Runs Deep: Implicit Reasoning Biases in Persona-Assigned LLMs
> Language models, like ChatGPT, can adopt different personas but this can inadvertently trigger biased reasoning patterns, leading to significant performance disparities across personas, highlighting the potential risks associated with persona-based interactions with LLMs.

<details>
<summary>Details</summary>
- **Abstract**: Recent work has showcased the ability of large-scale language models (LLMs) to embody diverse personas in their responses, exemplified by prompts like "_You are Julius Caesar. Compose a rap about Climate Change._" However, it remains unclear how these persona assignments indirectly influence LLMs\' core capabilities.  We present the first extensive study of this in the context of LLMs\' ability to perform basic reasoning. Our study encompasses 16 personas spanning 5 diverse groups (race, gender, religion, disability, and political affiliation), across 24 reasoning datasets in diverse domains such as mathematics, history, law, ethics, and more. Our findings unveil that while LLMs, such as ChatGPT, overtly reject stereotypes when explicitly asked ("_Are Black people inept at mathematics?_"), they tend to manifest implicit stereotypical and often erroneous presumptions when prompted to take on a persona (e.g., abstentions in rationales such as "_As a Black person, I am unable to answer this question as it requires math knowledge_"). This results in substantial disparities in reasoning performance among personas. This inherent \'deep\' bias permeates extensively, leading to a statistically significant performance drop in over 95% of our datasets for certain personas, with as much as 70% relative drop in accuracy on select datasets. Beyond explicit abstentions, these models also have implicitly biased reasoning not evident in their responses. We find that simple prompt-based mitigation approaches have minimal impact. Our findings serve as a cautionary tale that the practice of assigning personas to LLMs---a trend on the rise---can surface their deep-rooted biases and have unforeseeable and detrimental side-effects.
- **OpenReview**: https://openreview.net/pdf?id=kGteeZ18Ir
        
</details>

### MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback
> MINT is a new benchmark that evaluates large language models' ability to solve complex tasks through multiple interactions, using tools and receiving natural language feedback. By studying 20 LLMs, the study revealed that tools and feedback can improve performance, but single-turn performance alone is not a reliable indicator of multi-turn capabilities.

<details>
<summary>Details</summary>
- **Abstract**: To solve complex tasks, large language models (LLMs) often require multiple rounds of interactions with the user, sometimes assisted by external tools. However, current evaluation protocols often emphasize benchmark performance with single-turn exchanges, neglecting the nuanced interactions among the user, LLMs, and external tools, while also underestimating the importance of natural language feedback from users. These oversights contribute to discrepancies between research benchmark evaluations and real-world use cases. We introduce MINT, a benchmark that evaluates LLMs' ability to solve tasks with multi-turn interactions by (1) using tools and (2) leveraging natural language feedback. To ensure reproducibility, we provide an evaluation framework where LLMs can access tools by executing Python code and receive users' natural language feedback simulated by GPT-4. We repurpose a diverse set of established evaluation datasets focusing on reasoning, coding, and decision-making and carefully curate them into a compact subset for efficient evaluation. Our analysis of 20 open- and closed-source LLMs offers intriguing findings. (a) LLMs generally benefit from tools and language feedback, with performance gains (absolute, same below) of 1--8% for each turn of tool use and 2--17% with natural language feedback. (b) Better single-turn performance does not guarantee better multi-turn performance. (c) Surprisingly, on the LLMs evaluated, supervised instruction-finetuning (SIFT) and reinforcement learning from human feedback (RLHF) generally hurt multi-turn capabilities. We expect MINT can help measure progress and incentivize research in improving LLMs' capabilities in multi-turn interactions, especially for open-source communities where multi-turn human evaluation can be less accessible compared to commercial LLMs with a larger user base.
- **OpenReview**: https://openreview.net/pdf?id=jp3gWrMuIZ
        
</details>

### LLM Augmented LLMs: Expanding Capabilities through Composition
> CALM proposes a novel approach to enhance foundational language models (LLMs) using more specific models, enabling improved performance in specialized domains with minimal disruption to existing capabilities.

<details>
<summary>Details</summary>
- **Abstract**: Foundational models with billions of parameters which have been trained on large corpus of data have demonstrated non-trivial skills in a variety of domains. However, due to their monolithic structure, it is challenging and expensive to augment them or impart new skills. On the other hand, due to their adaptation abilities,several new instances of these models are being trained towards new domains and tasks.  In this work, we study the problem of efficient and practical composition of existing foundation models with more specific models to enable newer capabilities. To this end,  we propose CALMComposition to Augment Language Modelswhich introduces cross-attention between models to compose their representations and enable new capabilities. Salient features of CALM are: (i) Scales up LLMs on new tasks by re-using existing LLMs along with a few additional parameters and data, (ii) Existing model weights are kept intact, and hence preserves existing capabilities, and (iii) Applies to diverse domains and settings. We illustrate that augmenting PaLM2-S with a smaller model trained on low-resource languages results in an absolute improvement of up to 13% on tasks like translation into English and arithmetic reasoning for low-resource languages. Similarly,when PaLM2-S is augmented with a code-specific model, we see a relative improvement of 40% over the base model for code generation and explanation taskson-par with fully fine-tuned counterparts.
- **OpenReview**: https://openreview.net/pdf?id=jjA4O1vJRz
        
</details>

### Knowledge Fusion of Large Language Models
> This paper proposes a novel approach for merging existing pre-trained LLMs into a more capable model, using knowledge fusion to combine their strengths while addressing architectural differences. The research confirms that fused LLMs can surpass the performance of individual source LLMs, showcasing enhanced capabilities across various tasks.

<details>
<summary>Details</summary>
- **Abstract**: While training large language models (LLMs) from scratch can generate models with distinct functionalities and strengths, it comes at significant costs and may result in redundant capabilities. Alternatively, a cost-effective and compelling approach is to merge existing pre-trained LLMs into a more potent model. However, due to the varying architectures of these LLMs, directly blending their weights is impractical. In this paper, we introduce the notion of knowledge fusion for LLMs, aimed at combining the capabilities of existing LLMs and transferring them into a single LLM. By leveraging the generative distributions of source LLMs, we externalize their collective knowledge and unique strengths, thereby potentially elevating the capabilities of the target model beyond those of any individual source LLM. We validate our approach using three popular LLMs with different architecturesLlama-2, MPT, and OpenLLaMAacross various benchmarks and tasks. Our findings confirm that the fusion of LLMs can improve the performance of the target model across a range of capabilities such as reasoning, commonsense, and code generation.
- **OpenReview**: https://openreview.net/pdf?id=jiDsk12qcz
        
</details>

### MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning
> To address the limitations of LLMs in complex reasoning, researchers have created MuSR, a dataset of multistep narratives that challenge LLMs with real-world scenarios, allowing for the evaluation of LLM reasoning capabilities while highlighting areas for improvement.

<details>
<summary>Details</summary>
- **Abstract**: While large language models (LLMs) equipped with techniques like chain-of-thought prompting have demonstrated impressive capabilities, they still fall short in their ability to reason robustly in complex settings. However, evaluating LLM reasoning is challenging because system capabilities continue to grow while benchmark datasets for tasks like logical deduction have remained static. We introduce MuSR, a dataset for evaluating language models on multistep soft reasoning tasks specified in a natural language narrative. This dataset has two crucial features. First, it is created through a novel neurosymbolic synthetic-to-natural generation algorithm, enabling the construction of complex reasoning instances that challenge GPT-4 (e.g., murder mysteries roughly 1000 words in length) and which can be scaled further as more capable LLMs are released. Second, our data instances are free text narratives corresponding to real-world domains of reasoning; this makes it simultaneously much more challenging than other synthetically-crafted benchmarks while remaining realistic and tractable for human annotators to solve with high accuracy. We evaluate a range of LLMs and prompting techniques on this dataset and characterize the gaps that remain for techniques like chain-of-thought to perform robust reasoning.
- **OpenReview**: https://openreview.net/pdf?id=jenyYQzue1
        
</details>

### Lightweight Language Model Calibration for Open-ended Question Answering with Varied Answer Lengths
> Litcab proposes a lightweight method to calibrate large language models (LLMs) by predicting a bias term to adjust output probabilities, resulting in better-calibrated models with improved detection of hallucinations.

<details>
<summary>Details</summary>
- **Abstract**: A model is considered well-calibrated when its probability estimate aligns with the true likelihood of the output being correct. Calibrating large language models (LLMs) is crucial, as it plays a vital role in detecting and mitigating hallucinations, a common issue of LLMs, as well as building more trustworthy models.  Yet popular neural model calibration techniques are not well-suited for LLMs due to their lack of flexibility in discerning answer correctness and their high computational costs. For instance, post-processing methods, e.g., temperature scaling, are often unable to reorder the candidate generations. Moreover, training-based methods require fine-tuning the entire model, which becomes impractical due to the increasing sizes of modern LLMs. In this paper, we present Litcab, a lightweight calibration mechanism consisting of a single linear layer that takes as input the sentence representation and predicts a bias term, which is then added to the LM output logits. Litcab results with better-calibrated models, by only adding and training <2% of the original model parameters. For evaluation, we construct CaT, a benchmark consisting of six open-ended question-answering (QA) tasks, covering responses ranging from short phrases to paragraphs. We test Litcab with Llama2-7B, where it improves calibration across all tasks. We further conduct a comprehensive evaluation with multiple popular open-sourced LLMs from GPT and LLaMA families, yielding the following key findings:  (i) Larger models within the same family exhibit better calibration. (ii) GPT-family models show superior calibration compared to LLaMA, Llama2 and Vicuna models despite having much fewer parameters. (iii) Fine-tuning pretrained model (e.g., LLaMA) with samples of focused purpose (e.g., conversations) may lead to worse calibration, highlighting the importance of fine-tuning setups.
- **OpenReview**: https://openreview.net/pdf?id=jH67LHVOIO
        
</details>

### Language Models Represent Space and Time
> Modern LLMs such as Llama-2 may not just memorize patterns but may have learned a comprehensive understanding of fundamental dimensions like space and time, capturing their linear representations across scales and entity types, suggesting they possess a "world model" rather than just superficial knowledge.

<details>
<summary>Details</summary>
- **Abstract**: The capabilities of large language models (LLMs) have sparked debate over whether such systems just learn an enormous collection of superficial statistics or a coherent model of the data generating process---a world model. We find evidence for the latter by analyzing the learned representations of three spatial datasets (world, US, NYC places) and three temporal datasets (historical figures, artworks, news headlines) in the Llama-2 family of models. We discover that LLMs learn linear representations of space and time across multiple scales. These representations are robust to prompting variations and unified across different entity types (e.g. cities and landmarks). In addition, we identify individual ``space neurons'' and ``time neurons'' that reliably encode spatial and temporal coordinates. Our analysis demonstrates that modern LLMs acquire structured knowledge about fundamental dimensions such as space and time, supporting the view that they learn not merely superficial statistics, but literal world models.
- **OpenReview**: https://openreview.net/pdf?id=jE8xbmvFin
        
</details>

### Spoken Question Answering and Speech Continuation Using Spectrogram-Powered LLM
> This study presents a new technique for adapting large language models (LLMs) to comprehend and generate spoken language by combining speech encoding and end-to-end training. The model excels in preserving speaker identity and producing coherent responses, while retaining the LLM's knowledge and outperforming current models in spoken language understanding tasks.

<details>
<summary>Details</summary>
- **Abstract**: We present a novel approach to adapting pre-trained large language models (LLMs) to perform question answering (QA) and speech continuation. By endowing the LLM with a pre-trained speech encoder, our model becomes able to take speech inputs and generate speech outputs. The entire system is trained end-to-end and operates directly on spectrograms, simplifying our architecture. Key to our approach is a training objective that jointly supervises speech recognition, text continuation, and speech synthesis using only paired speech-text pairs, enabling a `cross-modal' chain-of-thought within a single decoding pass. Our method surpasses existing spoken language models in speaker preservation and semantic coherence. Furthermore, the proposed model improves upon direct initialization in retaining the knowledge of the original LLM as demonstrated through spoken QA datasets.
- **OpenReview**: https://openreview.net/pdf?id=izrOLJov5y
        
</details>

### Learning Performance-Improving Code Edits
> This paper investigates the potential of LLMs for high-level program optimization by leveraging a curated dataset of human-made edits and employing various adaptation strategies, ultimately yielding impressive performance improvements that exceed human abilities.

<details>
<summary>Details</summary>
- **Abstract**: With the waning of Moore\'s law, optimizing program performance has become a major focus of software research. However, high-level optimizations such as API and algorithm changes remain elusive due to the difficulty of understanding the semantics of code. Simultaneously, pretrained large language models (LLMs) have demonstrated strong capabilities at solving a wide range of programming tasks. To that end, we introduce a framework for adapting LLMs to high-level program optimization. First, we curate a dataset of performance-improving edits made by human programmers of over 77,000 competitive C++ programming submission pairs, accompanied by extensive unit tests. A major challenge is the significant variability of measuring performance on commodity hardware, which can lead to spurious "improvements". To isolate and reliably evaluate the impact of program optimizations, we design an environment based on the gem5 full system simulator, the de facto simulator used in academia and industry. Next, we propose a broad range of adaptation strategies for code optimization; for prompting, these include retrieval-based few-shot prompting and chain-of-thought, and for finetuning, these include performance-conditioned generation and synthetic data augmentation based on self-play. A combination of these techniques achieves an average speedup of 5.65 times on CodeLlama-13B and 6.86 times on GPT-3.5, surpassing the best human performance (4.06 times). We find our proposed performance-conditioned generation is particularly effective at improving performance as well as increasing the fraction of optimized programs.
- **OpenReview**: https://openreview.net/pdf?id=ix7rLVHXyY
        
</details>

### Causally Aligned Curriculum Learning
> This paper explores the use of causal reasoning to improve curriculum learning in Reinforcement Learning (RL), where the goal is to accelerate learning by training on a series of progressively more challenging tasks. The study derives conditions for identifying source tasks that share optimal decision rules with the target task, ensuring that skills learned from these tasks can be transferred to the target environment, even in the presence of unobserved confounders.

<details>
<summary>Details</summary>
- **Abstract**: A pervasive challenge in Reinforcement Learning (RL) is the ``curse of dimensionality'' which is the exponential growth in the state-action space when optimizing a high-dimensional target task (Bellman, 95). The framework of curriculum learning trains the agent in a curriculum composed of a sequence of related and more manageable source tasks. The expectation is that when some optimal decision rules are shared across source tasks and the target task, the agent could more quickly pick up the necessary skills to behave optimally in the environment, thus accelerating the learning process.  However, this critical assumption of invariant optimal decision rules does not necessarily hold in many practical applications, specifically when the underlying environment contains unobserved confounders. This paper studies the problem of curriculum RL through causal lenses. We derive a sufficient graphical condition characterizing causally aligned source tasks, i.e., the invariance of optimal decision rules holds. We further develop an efficient algorithm to generate a causally aligned curriculum, provided with qualitative causal knowledge of the target environment. Finally, we validate our proposed methodology through experiments in confounded environments.
- **OpenReview**: https://openreview.net/pdf?id=hp4yOjhwTs
        
</details>

### Generating Images in Context with Multimodal Large Language Models
> Kosmos-G, leveraging advanced perception capabilities of Multimodal Large Language Models and compositional instruction tuning, showcases a distinctive ability to generate images from generalized vision-language inputs, including multiple images, without requiring modifications to the image decoder.

<details>
<summary>Details</summary>
- **Abstract**: Recent advancements in text-to-image (T2I) and vision-language-to-image (VL2I) generation have made significant strides. However, the generation from generalized vision-language inputs, especially involving multiple images, remains under-explored. This paper presents Kosmos-G, a model that leverages the advanced perception capabilities of Multimodal Large Language Models (MLLMs) to tackle the aforementioned challenge. Our approach aligns the output space of MLLM with CLIP using the textual modality as an anchor and performs compositional instruction tuning on curated data. Kosmos-G demonstrates a unique capability of zero-shot multi-entity subject-driven generation. Notably, the score distillation instruction tuning requires no modifications to the image decoder. This allows for a seamless substitution of CLIP and effortless integration with a myriad of U-Net techniques ranging from fine-grained controls to personalized image decoder variants. We posit Kosmos-G as an initial attempt towards the goal of ``image as a foreign language in image generation.
- **OpenReview**: https://openreview.net/pdf?id=he6mX9LTyE
        
</details>

### Bongard-OpenWorld: Few-Shot Reasoning for Free-form Visual Concepts in the Real World
> "Bongard-OpenWorld" is a complex benchmark that assesses AI\'s ability to grasp novel visual concepts from a limited set of examples using real-world images. Despite the use of sophisticated language and vision models, current AI systems struggle to match human performance on this task.

<details>
<summary>Details</summary>
- **Abstract**: We introduce $\\textbf{Bongard-OpenWorld}$, a new benchmark for evaluating real-world few-shot reasoning for machine vision. It originates from the classical Bongard Problems (BPs): Given two sets of images (positive and negative), the model needs to identify the set that query images belong to by inducing the visual concept, which is exclusively depicted by images from the positive set. Our benchmark inherits the few-shot concept induction of the original BPs while adding the two novel layers of challenge: 1) open-world free-form concepts, as the visual concepts in Bongard-OpenWorld are unique compositions of terms from an open vocabulary, ranging from object categories to abstract visual attributes and commonsense factual knowledge; 2)  real-world images, as opposed to the synthetic diagrams used by many counterparts. In our exploration, Bongard-OpenWorld already imposes a significant challenge to current few-shot reasoning algorithms. We further investigate to which extent the recently introduced Large Language Models (LLMs) and Vision-Language Models (VLMs) can solve our task, by directly probing VLMs, and combining VLMs and LLMs in an interactive reasoning scheme. We even designed a neuro-symbolic reasoning approach that reconciles LLMs & VLMs with logical reasoning to emulate the human problem-solving process for Bongard problems. However, none of these approaches manage to close the human-machine gap, as the best learner achieves 64% accuracy while human participants easily reach 91%. We hope Bongard-OpenWorld can help us better understand the limitations of current visual intelligence and facilitate future research on visual agents with stronger few-shot visual reasoning capabilities. All implementation details and reproduction code, including Bongard-OpenWorld dataset, are available in an anonymous github repository https://github.com/Bongard-OpenWorld.
- **OpenReview**: https://openreview.net/pdf?id=hWS4MueyzC
        
</details>

### Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!
> Fine-tuning large language models (LLMs) with user-specific data can compromise their safety alignment, even with a small number of adversarial training examples. This highlights a gap in existing safety protocols, suggesting the need for robust measures to maintain model safety in customized fine-tuning environments.

<details>
<summary>Details</summary>
- **Abstract**: Optimizing large language models (LLMs) for downstream use cases often involves the customization of pre-trained LLMs through further fine-tuning. Meta's open-source release of Llama models and OpenAI's APIs for fine-tuning GPT-3.5 Turbo on customized datasets accelerate this trend. But, what are the safety costs associated with such customized fine-tuning? While existing safety alignment techniques restrict harmful behaviors of LLMs at inference time, they do not cover safety risks when fine-tuning privileges are extended to end-users. Our red teaming studies find that the safety alignment of LLMs can be compromised by fine-tuning with only a few adversarially designed training examples. For instance, we jailbreak GPT-3.5 Turbo's safety guardrails by fine-tuning it on only 10 such examples at a cost of less than $0.20 via OpenAI's APIs, making the model responsive to nearly any harmful instructions. Disconcertingly, our research also reveals that, even without malicious intent, simply fine-tuning with benign and commonly used datasets can also inadvertently degrade the safety alignment of LLMs, though to a lesser extent. These findings suggest that fine-tuning aligned LLMs introduces new safety risks that current safety infrastructures fall short of addressing --- even if a model's initial safety alignment is impeccable, how can it be maintained after customized fine-tuning? We outline and critically analyze potential mitigations and advocate for further research efforts toward reinforcing safety protocols for the customized fine-tuning of aligned LLMs.  (This paper contains red-teaming data and model-generated content that can be offensive in nature.)"
- **OpenReview**: https://openreview.net/pdf?id=hTEGyKf0dZ
        
</details>

### Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
> Self-RAG, a retrieval-augmented LM framework, reduces LM hallucination by enabling self-reflection during inference, allowing it to retrieve relevant passages on-demand and adapt its generations based on the retrieved information.

<details>
<summary>Details</summary>
- **Abstract**: Retrieval-Augmented Generation (RAG), an ad hoc approach that augments Language Models (LMs)  with retrieval, decreases hallucination issues of large LMs. However, indiscriminately retrieving and incorporating a fixed number of retrieved passages, regardless of whether retrieval is necessary, or passages are relevant, diminishes LM versatility or can lead to unhelpful response generation. In this work, we introduce a new framework called **Self-Reflective Retrieval-Augmented Generation (Self-RAG)** that enhances an LM's quality and factuality through retrieval and self-reflection.   Our framework trains a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and its own generations using special tokens, called *reflection* tokens. Generating reflection tokens makes the LM controllable during the inference phase, enabling it to tailor its behavior to diverse task requirements.  Experiments show that Self-RAG (7B and 13B parameters) significantly outperforms state-of-the-art LLMs and retrieval-augmented models on a diverse set of tasks.  Specifically, Self-RAG outperforms ChatGPT and retrieval-augmented Llama2-chat on multiple tasks including Open-domain QA and fact verification, and it shows significant gains in factuality scores and citation accuracy for long-form generations relative to these models.
- **OpenReview**: https://openreview.net/pdf?id=hSyW5go0v8
        
</details>

### Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks
> This research combines language learning models with reinforcement learning to guide robots in complex tasks, without requiring predefined skills. The proposed approach, Plan-Seq-Learn, uses motion planning to translate abstract language instructions into low-level control actions, enabling robots to effectively solve long-horizon tasks with over 80% success rate.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) are highly capable of performing planning for long-horizon robotics tasks, yet existing methods require access to a pre-defined skill library (*e.g.* picking, placing, pulling, pushing, navigating). However, LLM planning does not address how to design or learn those behaviors, which remains challenging particularly in long-horizon settings. Furthermore, for many tasks of interest, the robot needs to be able to adjust its behavior in a fine-grained manner, requiring the agent to be capable of modifying *low-level* control actions. Can we instead use the internet-scale knowledge from LLMs for high-level policies, guiding reinforcement learning (RL) policies to efficiently solve robotic control tasks online without requiring a pre-determined set of skills? In this paper, we propose **Plan-Seq-Learn** (PSL): a modular approach that uses motion planning to bridge the gap between abstract language and learned low-level control for solving long-horizon robotics tasks from scratch. We demonstrate that PSL is capable of solving 20+ challenging single and multi-stage robotics tasks on four benchmarks at success rates of over 80% from raw visual input, out-performing language-based, classical, and end-to-end approaches. Video results and code at https://planseqlearn.github.io/'
- **OpenReview**: https://openreview.net/pdf?id=hQVCCxQrYN
        
</details>

### True Knowledge Comes from Practice: Aligning Large Language Models with Embodied Environments via Reinforcement Learning
> TWOSOME merges LLMs and RL by utilizing LLMs for decision-making and aligning them with environments through RL, achieving superior sample efficiency and performance in various tasks while preserving LLMs' open-vocabulary capabilities.

<details>
<summary>Details</summary>
- **Abstract**: Despite the impressive performance across numerous tasks, large language models (LLMs) often fail in solving simple decision-making tasks due to the misalignment of the knowledge in LLMs with environments. On the contrary, reinforcement learning (RL) agents learn policies from scratch, which makes them always align with environments but difficult to incorporate prior knowledge for efficient explorations. To narrow the gap, we propose TWOSOME, a novel general online framework that deploys LLMs as decision-making agents to efficiently interact and align with embodied environments via RL without requiring any prepared datasets or prior knowledge of the environments. Firstly, we query the joint probabilities of each valid action with LLMs to form behavior policies. Then, to enhance the stability and robustness of the policies, we propose two normalization methods and summarize four prompt design principles. Finally, we design a novel parameter-efficient training architecture where the actor and critic share one frozen LLM equipped with low-rank adapters (LoRA) updated by PPO. We conduct extensive experiments to evaluate TWOSOME. i) TWOSOME exhibits significantly better sample efficiency and performance compared to the conventional RL method, PPO, and prompt tuning method, SayCan, in both classical decision-making environment, Overcooked, and simulated household environment, VirtualHome. ii) Benefiting from LLMs' open-vocabulary feature, TWOSOME shows superior generalization ability to unseen tasks. iii) Under our framework, there is no significant loss of the LLMs' original ability during online PPO finetuning.
- **OpenReview**: https://openreview.net/pdf?id=hILVmJ4Uvu
        
</details>

### Label-free Node Classification on Graphs with Large Language Models (LLMs)
> This work proposes a novel pipeline, LLM-GNN, that combines the strengths of GNNs and LLMs for label-free node classification. By leveraging LLMs to annotate a small fraction of nodes and then training GNNs on these annotations, LLM-GNN efficiently handles structural data, reduces inference costs, and achieves promising accuracy, as demonstrated by experiments on a large-scale dataset.

<details>
<summary>Details</summary>
- **Abstract**: In recent years, there have been remarkable advancements in node classification achieved by Graph Neural Networks (GNNs). However, they necessitate abundant high-quality labels to ensure promising performance. In contrast, Large Language Models (LLMs) exhibit impressive zero-shot proficiency on text-attributed graphs. Yet, they face challenges in efficiently processing structural data and suffer from high inference costs. In light of these observations, this work introduces a label-free node classification on graphs with LLMs pipeline, LLM-GNN. It amalgamates the strengths of both GNNs and LLMs while mitigating their limitations. Specifically, LLMs are leveraged to annotate a small portion of nodes and then GNNs are trained on LLMs' annotations to make predictions for the remaining large portion of nodes. The implementation of LLM-GNN faces a unique challenge: how can we actively select nodes for LLMs to annotate and consequently enhance the GNN training? How can we leverage LLMs to obtain annotations of high quality, representativeness, and diversity, thereby enhancing GNN performance with less cost? To tackle this challenge, we develop an annotation quality heuristic and leverage the confidence scores derived from LLMs to advanced node selection. Comprehensive experimental results validate the effectiveness of LLM-GNN. In particular, LLM-GNN can achieve an accuracy of 74.9% on a vast-scale dataset \\products with a cost less than 1 dollar.
- **OpenReview**: https://openreview.net/pdf?id=hESD2NJFg8
        
</details>

### Language Model Beats Diffusion - Tokenizer is key to visual generation
> Our proposed visual tokenizer, \\modelname{}, enables Large Language Models to surpass diffusion models on image and video generation, with applications in video compression and action recognition.

<details>
<summary>Details</summary>
- **Abstract**: While Large Language Models (LLMs) are the dominant models for generative tasks in language, they do not perform as well as diffusion models on image and video generation. To effectively use LLMs for visual generation, one crucial component is the visual tokenizer that maps pixel-space inputs to discrete tokens appropriate for LLM learning. In this paper, we introduce \\modelname{}, a video tokenizer designed to generate concise and expressive tokens for both videos and images using a common token vocabulary. Equipped with this new tokenizer, we show that LLMs outperform diffusion models on standard image and video generation benchmarks including ImageNet and Kinetics. In addition, we demonstrate that our tokenizer surpasses the previously top-performing video tokenizer on two more tasks: (1) video compression comparable to the next-generation video codec (VCC) according to human evaluations, and (2) learning effective representations for action recognition tasks.
- **OpenReview**: https://openreview.net/pdf?id=gzqrANCF4g
        
</details>

### Generative Judge for Evaluating Alignment
> **Auto-J**, a 13B-parameter generative judge, is proposed to evaluate large language models' generality, flexibility, and interpretability in natural language processing tasks by scrutinizing responses with concise explanations. Extensive testing on diverse scenarios demonstrates its superiority in evaluating models, opening new avenues for model assessment and development.

<details>
<summary>Details</summary>
- **Abstract**: The rapid development of Large Language Models (LLMs) has substantially expanded the range of tasks they can address. In the field of Natural Language Processing (NLP), researchers have shifted their focus from conventional NLP tasks (e.g., sequence tagging and parsing) towards tasks that revolve around aligning with human needs (e.g., brainstorming and email writing). This shift in task distribution imposes new requirements on evaluating these aligned models regarding *generality* (i.e., assessing performance across diverse scenarios), *flexibility* (i.e., examining under different protocols), and *interpretability* (i.e., scrutinizing models with explanations). In this paper, we propose a generative judge with 13B parameters, **Auto-J**, designed to address these challenges. Our model is trained on user queries and LLM-generated responses under massive real-world scenarios and accommodates diverse evaluation protocols (e.g., pairwise response comparison and single-response evaluation) with well-structured natural language critiques. To demonstrate the efficacy of our approach, we construct a new testbed covering 58 different scenarios. Experimentally, **Auto-J** outperforms a series of strong competitors, including both open-source and closed-source models, by a large margin. We also provide detailed analysis and case studies to further reveal the potential of our method and make a variety of resources public at https://anonymous.4open.science/r/Auto-J-ICLR-ver-0107.
- **OpenReview**: https://openreview.net/pdf?id=gtkFw6sZGS
        
</details>

### Can LLMs Keep a Secret? Testing  Privacy  Implications of Language Models  via Contextual Integrity Theory
> LLMs struggle to protect privacy in interactive settings due to their limited ability to discern appropriate information sharing based on context, with leading models disclosing private data significantly more often than humans.

<details>
<summary>Details</summary>
- **Abstract**: Existing efforts on quantifying privacy implications for large language models (LLMs) solely focus on measuring leakage of training data. In this work, we shed light on the often-overlooked interactive settings where an LLM receives information from multiple sources and generates an output to be shared with other entities, creating the potential of exposing sensitive input data in inappropriate contexts. In these scenarios, humans nat- urally uphold privacy by choosing whether or not to disclose information depending on the context. We ask the question Can LLMs demonstrate an equivalent discernment and reasoning capability when considering privacy in context? We propose CONFAIDE, a benchmark grounded in the theory of contextual integrity and designed to identify critical weaknesses in the privacy reasoning capabilities of instruction-tuned LLMs. CONFAIDE consists of four tiers, gradually increasing in complexity, with the final tier evaluating contextual privacy reasoning and theory of mind capabilities. Our experiments show that even commercial models such as GPT-4 and ChatGPT reveal private information in contexts that humans would not, 39% and 57% of the time, respectively, highlighting the urgent need for a new direction of privacy-preserving approaches as we demonstrate a larger underlying problem stemmed in the models lack of reasoning capabilities.
- **OpenReview**: https://openreview.net/pdf?id=gmg7t8b4s0
        
</details>

### Confronting Reward Model Overoptimization with Constrained RLHF
> The study investigates the impact of composite reward models on overoptimization in large language models and introduces a constrained reinforcement learning approach to overcome this issue by dynamically weighting component models, leading to improved evaluation performance.

<details>
<summary>Details</summary>
- **Abstract**: Large language models are typically aligned with human preferences by optimizing reward models (RMs) fitted to human feedback. However, human preferences are multi-faceted, and it is increasingly common to derive reward from a composition of simpler reward models which each capture a different aspect of language quality. This itself presents a challenge, as it is difficult to appropriately weight these component RMs when combining them. Compounding this difficulty, because any RM is only a proxy for human evaluation, this process is vulnerable to *overoptimization*, wherein past a certain point, accumulating higher reward is associated with worse human ratings. In this paper, we perform the first study on overoptimization in composite RMs, showing that correlation between component RMs has a significant effect on the locations of these points. We then introduce an approach to solve this issue using constrained reinforcement learning as a means of preventing the agent from exceeding each RM's threshold of usefulness. Our method addresses the problem of weighting component RMs by learning dynamic weights, naturally given by the Lagrange multipliers. As a result, each RM stays within the range at which it is an effective proxy, improving evaluation performance. Finally, we introduce an adaptive method using gradient-free optimization to identify and optimize towards these points during a single run.
- **OpenReview**: https://openreview.net/pdf?id=gkfUvn0fLU
        
</details>

### DyVal: Graph-informed Dynamic Evaluation of Large Language Models
> DyVal, a novel evaluation protocol, dynamically generates evaluation samples with controllable complexities, revealing the inadequacy of static benchmarks in assessing the capabilities of large language models (LLMs).

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have achieved remarkable performance in various evaluation benchmarks. However, concerns about their performance are raised on potential data contamination in their considerable volume of training corpus. Moreover, the static nature and fixed complexity of current benchmarks may inadequately gauge the advancing capabilities of LLMs. In this paper, we introduce DyVal, a novel, general, and flexible evaluation protocol for dynamic evaluation of LLMs. Based on our proposed dynamic evaluation framework, we build graph-informed DyVal by leveraging the structural advantage of directed acyclic graphs to dynamically generate evaluation samples with controllable complexities. DyVal generates challenging evaluation sets on reasoning tasks including mathematics, logical reasoning, and algorithm problems. We evaluate various LLMs ranging from Flan-T5-large to ChatGPT and GPT4. Experiments demonstrate that LLMs perform worse in DyVal-generated evaluation samples with different complexities, emphasizing the significance of dynamic evaluation. We also analyze the failure cases and results of different prompting methods. Moreover, DyVal-generated samples are not only evaluation sets, but also helpful data for fine-tuning to improve the performance of LLMs on existing benchmarks. We hope that DyVal can shed light on the future evaluation research of LLMs.
- **OpenReview**: https://openreview.net/pdf?id=gjfOL9z5Xr
        
</details>

### Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs
> The paper proposes a novel framework for eliciting confidence from closed-source LLMs, exploring various prompt and aggregation techniques to improve calibration and failure prediction performance. Advanced LLMs exhibit overconfidence like humans, but their performance varies with task complexity and aggregation strategies, demonstrating the need for ongoing research in this area.

<details>
<summary>Details</summary>
- **Abstract**: Empowering large language models (LLMs) to accurately express confidence in their answers is essential for reliable and trustworthy decision-making. Previous confidence elicitation methods, which primarily rely on *white-box access* to internal model information or model fine-tuning, have become less suitable for LLMs, especially closed-source commercial APIs. This leads to a growing need to explore the untapped area of *black-box* approaches for LLM uncertainty estimation. To better break down the problem, we define a systematic framework with three components: *prompting* strategies for eliciting verbalized confidence, *sampling* methods for generating multiple responses, and *aggregation* techniques for computing consistency. We then benchmark these methods on two key tasksconfidence calibration and failure predictionacross five types of datasets (e.g., commonsense and arithmetic reasoning) and five widely-used LLMs including GPT-4 and LLaMA 2.  Our analysis uncovers several key insights: 1) LLMs, when verbalizing their confidence, tend to be overconfident, potentially imitating human patterns of expressing confidence. 2) As model capability scales up, both calibration and failure prediction performance improve, yet still far from ideal performance. 3) Human-inspired prompting strategies mitigate this overconfidence, albeit with diminishing returns in advanced models like GPT-4, especially in improving failure prediction. 4) Employing sampling strategies paired with specific aggregators can effectively enhance failure prediction; moreover, the choice of aggregator can be tailored based on the desired performance enhancement. Despite these advancements, all investigated methods struggle in challenging tasks, such as those requiring professional knowledge, indicating significant scope for improvement. We believe this study can serve as a strong baseline and provide insights for eliciting confidence in black-box LLMs.
- **OpenReview**: https://openreview.net/pdf?id=gjeQKFxFpZ
        
</details>

### Attention Satisfies: A Constraint-Satisfaction Lens on Factual Errors of Language Models
> By examining how Transformer-based LLMs handle factual constraints, researchers discovered a link between attention to constraints and accuracy in generated text, suggesting that attention patterns can predict factual errors and improve the reliability of LLMs.

<details>
<summary>Details</summary>
- **Abstract**: We investigate the internal behavior of Transformer-based Large Language Models (LLMs) when they generate factually incorrect text. We propose modeling factual queries as constraint satisfaction problems and use this framework to investigate how the LLM interacts internally with factual constraints. We find a strong positive relationship between the LLM's attention to constraint tokens and the factual accuracy of generations. We curate a suite of 11 datasets containing over 40,000 prompts to study the task of predicting factual errors with the Llama-2 family across all scales (7B, 13B, 70B). We propose SAT Probe, a method probing attention patterns, that can predict factual errors and fine-grained constraint satisfaction, and allow early error identification. The approach and findings take another step towards using the mechanistic understanding of LLMs to enhance their reliability.
- **OpenReview**: https://openreview.net/pdf?id=gfFVATffPd
        
</details>

### Safety-Tuned LLaMAs: Lessons From Improving the Safety of Large Language Models that Follow Instructions
> Instructing large language models to prioritize helpfulness enhances their performance but raises safety concerns, as demonstrated by the vulnerability of current models to malicious instructions and harm generation. The research suggests that incorporating a small number of safety examples during fine-tuning can significantly improve model safety, potentially leading to trade-offs between helpfulness and safety.

<details>
<summary>Details</summary>
- **Abstract**: Training large language models to follow instructions makes them perform better on a wide range of tasks and generally become more helpful. However, a perfectly helpful model will follow even the most malicious instructions and readily generate harmful content. In this paper, we raise concerns over the safety of models that only emphasize helpfulness, not harmlessness, in their instruction-tuning. We show that several popular instruction-tuned models are highly unsafe. Moreover, we show that adding just 3% safety examples (a few hundred demonstrations) when fine-tuning a model like LLaMA can substantially improve its safety. Our safety-tuning does not make models significantly less capable or helpful as measured by standard benchmarks. However, we do find exaggerated safety behaviours, where too much safety-tuning makes models refuse perfectly safe prompts if they superficially resemble unsafe ones. As a whole, our results illustrate trade-offs in training LLMs to be helpful and training them to be safe.
- **OpenReview**: https://openreview.net/pdf?id=gT5hALch9z
        
</details>

### A Private Watermark for Large Language Models
> This study presents a private text watermarking algorithm that uses distinct neural networks for watermark generation and detection, eliminating the need for a shared secret key and improving security against counterfeiting while maintaining high detection accuracy and computational efficiency.

<details>
<summary>Details</summary>
- **Abstract**: Recently, text watermarking algorithms for large language models (LLMs) have been proposed to mitigate the potential harms of text generated by LLMs, including fake news and copyright issues. However, current watermark detection algorithms require the secret key used in the watermark generation process, making them susceptible to security breaches and counterfeiting. To address this limitation, we propose the first private watermarking algorithm that uses two different neural networks for watermark generation and detection, instead of using the same key at both stages. Meanwhile, the token embedding parameters are shared between the generation and detection networks, which makes the detection network achieve a high accuracy very efficiently. Experiments demonstrate that Our algorithm attains high detection accuracy and computational efficiency through neural networks with a minimized number of parameters. Subsequent analysis confirms the high complexity involved in reverse-engineering the watermark generation algorithms from the detection network.
- **OpenReview**: https://openreview.net/pdf?id=gMLQwKDY3N
        
</details>

### BESA: Pruning Large Language Models with Blockwise Parameter-Efficient Sparsity Allocation
> This paper proposes BESA, a novel LLM pruning technique that outperforms existing methods by targeting overall pruning error with respect to transformer blocks and allocating layer-specific sparsity in a differentiable manner, resulting in state-of-the-art performance and efficient LLM pruning.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have demonstrated outstanding performance in various tasks, such as text summarization, text question-answering, and etc. While their performance is impressive, the computational footprint due to their vast number of parameters can be prohibitive. Existing solutions such as SparseGPT and Wanda attempt to alleviate this issue through weight pruning. However, their layer-wise approach results in significant perturbation to the model's output and requires meticulous hyperparameter tuning, such as the pruning rate, which can adversely affect overall model performance. To address this, this paper introduces a novel LLM pruning technique dubbed blockwise parameter-efficient sparsity allocation (BESA) by applying a blockwise reconstruction loss. In contrast to the typical layer-wise pruning techniques, BESA is characterized by two distinctive attributes: i) it targets the overall pruning error with respect to individual transformer blocks, and ii) it allocates layer-specific sparsity in a differentiable manner, both of which ensure reduced performance degradation after pruning. Our experiments show that BESA achieves state-of-the-art performance, efficiently pruning LLMs like LLaMA1, and LLaMA2 with 7B to 70B parameters on a single A100 GPU in just five hours. Code is available at [here](https://github.com/LinkAnonymous/BESA).
- **OpenReview**: https://openreview.net/pdf?id=gC6JTEU3jl
        
</details>

### Evaluating the Zero-shot Robustness of Instruction-tuned Language Models
> Instruction fine-tuning for language models shows promise for zero-shot learning but lacks robustness to instruction phrasings. This paper highlights the performance degradation caused by novel instruction phrasings and proposes "soft prompt" embeddings to enhance robustness against language variations.

<details>
<summary>Details</summary>
- **Abstract**: Instruction fine-tuning has recently emerged as a promising approach for improving the zero-shot capabilities of Large Language Models (LLMs) on new tasks. This technique has shown particular strength in improving the performance of modestly sized LLMs, sometimes inducing performance competitive with much larger model variants. In this paper, we ask two questions: (1) How sensitive are instruction-tuned models to the particular phrasings of instructions, and, (2) How can we make them more robust to such natural language variation? To answer the former, we collect a set of 319 instructions manually written by NLP practitioners for over 80 unique tasks included in widely used benchmarks, and we evaluate the variance and average performance of these instructions as compared to instruction phrasings observed during instruction fine-tuning. We find that using novel (unobserved) but appropriate instruction phrasings consistently degrades model performance, sometimes substantially so. Further, such natural instructions yield a wide variance in downstream performance, despite their semantic equivalence. Put another way, instruction-tuned models are not especially robust to instruction re-phrasings.  We propose a simple method to mitigate this issue by introducing ``soft prompt'' embedding parameters and optimizing these to maximize the similarity between representations of semantically equivalent instructions. We show that this method consistently improves the robustness of instruction-tuned models.
- **OpenReview**: https://openreview.net/pdf?id=g9diuvxN6D
        
</details>

### Improving Generalization of Alignment with Human Preferences through Group Invariant Learning
> To enhance the performance of AI assistants, researchers introduce a technique that enables the model to consistently perform well across different domains by deliberately focusing on challenging data groups and adaptively adjusting the exploration space, resulting in improved stability and generalization.

<details>
<summary>Details</summary>
- **Abstract**: The success of AI assistants based on language models (LLMs) hinges crucially on Reinforcement Learning from Human Feedback (RLHF), which enables the generation of responses more aligned with human preferences.  As universal AI assistants, there's a growing expectation for them to perform consistently across various domains.  However, previous work shows that Reinforcement Learning (RL) often exploits shortcuts to attain high rewards and overlooks challenging samples. This focus on quick reward gains undermines both the stability in training and the model's ability to generalize to new, unseen data. In this work, we propose a novel approach that can learn a consistent policy via RL across various data groups or domains.  Given the challenges associated with acquiring group annotations, our method automatically classifies data into different groups, deliberately maximizing performance variance. Then, we optimize the policy to perform well on challenging groups.  Lastly, leveraging the established groups, our approach adaptively adjusts the exploration space, allocating more learning capacity to more challenging data and preventing the model from over-optimizing on simpler data. Experimental results indicate that our approach significantly enhances training stability and model generalization.
- **OpenReview**: https://openreview.net/pdf?id=fwCoLe3TAX
        
</details>

### Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game
> This study presents a substantial dataset of adversarial prompts crafted by players of an online game, highlighting the vulnerabilities of instruction-following LLMs to prompt injection attacks. It establishes benchmarks for resistance to these attacks and provides insights into the underlying weaknesses of such models.

<details>
<summary>Details</summary>
- **Abstract**: We present a dataset of over 100,000 prompt injection attacks and 30,000 anti-injection "defense" prompts created by players of an online game, Tensor Trust. To the best of our knowledge, it is the largest dataset of human-generated adversarial examples for instruction-following LLMs. Using the Tensor Trust dataset, we create benchmarks for resistance to two types of prompt injection (which we refer to as prompt extraction and prompt hijacking) as well as a benchmark for detecting when an LLM has leaked sensitive information from the prompt. We also show that many attacks in our dataset have an intuitive structure that sheds light on the weaknesses of these models. The full Tensor Trust dataset and source code are available at `[URL removed for review]`.
- **OpenReview**: https://openreview.net/pdf?id=fsW7wJGLBd
        
</details>

### Are Human-generated Demonstrations Necessary for In-context Learning?
> Self-contemplation prompting, an approach that frees large language models from human-generated demonstrations in learning, rivals the performance of traditional methods that rely on external data.

<details>
<summary>Details</summary>
- **Abstract**: Despite the promising few-shot ability of large language models (LLMs), the standard paradigm of In-context Learning (ICL) suffers the disadvantages of susceptibility to selected demonstrations and the intricacy to generate these demonstrations. In this paper, we raise the fundamental question that whether human-generated demonstrations are necessary for ICL. To answer this question, we propose self-contemplation prompting strategy (SEC), a paradigm free from human-crafted demonstrations. The key point of SEC is that, instead of using hand-crafted examples as demonstrations in ICL, SEC asks LLMs to first create demonstrations on their own, based on which the final output is generated. SEC is a flexible framework and can be adapted to both the vanilla ICL and the chain-of-thought (CoT), but with greater ease: as the manual-generation process of both examples and rationale can be saved. Extensive experiments in arithmetic reasoning, commonsense reasoning, multi-task language understanding, and code generation benchmarks, show that SEC, which does not require hand-crafted demonstrations, significantly outperforms the zero-shot learning strategy, and achieves comparable results to ICL with hand-crafted demonstrations. This demonstrates that, for many tasks, contemporary LLMs possess a sufficient level of competence to exclusively depend on their own capacity for decision making, removing the need for external training data.
- **OpenReview**: https://openreview.net/pdf?id=frRDT6EOhg
        
</details>

### Sample-efficient Learning of Infinite-horizon Average-reward MDPs with General Function Approximation
> This paper combines model-based and value-based approaches in the FLOP framework, designed for average-reward Markov decision processes with general function approximation. FLOP introduces a novel confidence set construction and policy updating scheme, resulting in a sublinear regret bound that captures exploration challenges in these processes.

<details>
<summary>Details</summary>
- **Abstract**: We study infinite-horizon average-reward Markov decision processes (AMDPs) in the context of general function approximation. Specifically, we propose a novel algorithmic framework named Fixed-Point Local Optimization (FLOP), which incorporates both model-based and value-based incarnations. In particular, FLOP features a novel construction of confidence sets and a low-switching policy updating scheme, which are tailored to the average-reward and function approximation setting. Moreover, for AMDPs, we propose a novel complexity measure --- average-reward generalized eluder coefficient (AGEC) --- which captures the challenge of exploration in AMDPs with general function approximation. Such a complexity measure encompasses almost all previously known tractable AMDP models, such as linear AMDPs and linear mixture AMDPs, and also includes newly identified cases such as kernel AMDPs and AMDPs with low Bellman eluder dimensions. Using AGEC, we prove that FLOP achieves a sublinear  $\\tilde{\\mathcal{O}}(\\mathrm{poly}(d, \\mathrm{sp}(v^*)) \\sqrt{T \\beta })$ regret, where $d$  and  $\\beta$ correspond to  AGEC and the log-covering number of the hypothesis class respectively,  $\\mathrm{sp}(v^*)$ represents the span of the optimal state bias function, $T$ denotes the number of steps, and $\\tilde{\\mathcal{O}} (\\cdot) $ omits logarithmic factors.  When specialized to concrete AMDP models, our regret bounds are comparable to those established by the existing algorithms designed specifically for these special cases.  To the best of our knowledge, this paper presents the first comprehensive theoretical framework capable of handling nearly all AMDPs.
- **OpenReview**: https://openreview.net/pdf?id=fq1wNrC2ai
        
</details>

### GAIA: a benchmark for General AI Assistants
> GAIA, a benchmark for AI assistants, evaluates their ability to tackle real-world questions requiring reasoning and tool proficiency. While humans excel at solving these questions, most AI systems struggle, highlighting the need for AGI systems to exhibit human-like performance on such tasks.

<details>
<summary>Details</summary>
- **Abstract**: We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. Our questions allow simple, fast, and factual verification. GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on  tasks requiring professional skills in e.g. law or chemistry. GAIA's philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system's capability to exhibit similar robustness as the average human does on such questions. Using GAIA's methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board \\href{https://huggingface.co/xxx}{hereby accessible}.
- **OpenReview**: https://openreview.net/pdf?id=fibxvahvs3
        
</details>

### A Paradigm Shift in Machine Translation: Boosting Translation Performance of Large Language Models
> This study presents a new fine-tuning strategy for LLMs, called ALMA, which enables superior translation performance with a minimal need for training data compared to existing LLM translation models.

<details>
<summary>Details</summary>
- **Abstract**: Generative Large Language Models (LLMs) have achieved remarkable advancements in various NLP tasks. However, these advances have not been reflected in the translation task, especially those with moderate model sizes (i.e., 7B or 13B parameters), which still lag behind conventional supervised encoder-decoder translation models. Previous studies have attempted to improve the translation capabilities of these LLMs, but their gains have been limited. In this study, we propose a novel fine-tuning approach for LLMs that is specifically designed for the translation task, eliminating the need for the abundant parallel data that traditional translation models usually depend on. Our approach consists of two fine-tuning stages: initial fine-tuning on monolingual data followed by subsequent fine-tuning on a small set of high-quality parallel data.  We introduce the LLM  developed through this strategy as **A**dvanced **L**anguage **M**odel-based tr**A**nslator (**ALMA**). Based on LLaMA-2 as our underlying model, our results show that the model can achieve an average improvement of more than 12 BLEU and 12 COMET over its zero-shot performance across 10 translation directions from the WMT'21 (2 directions) and WMT'22 (8 directions) test datasets. The performance is significantly better than all prior work and even superior to the NLLB-54B model \\citep{nllb} and GPT-3.5-text-davinci-003, with only 7B or 13B parameters. This method establishes the foundation for a novel training paradigm in machine translation.
- **OpenReview**: https://openreview.net/pdf?id=farT6XXntP
        
</details>

### Unveiling the Pitfalls of Knowledge Editing for Large Language Models
> This paper investigates potential risks of knowledge editing in LLMs, identifying "Knowledge Conflict" and "Knowledge Distortion" as concerns. It demonstrates that editing knowledge can introduce inconsistencies and warp the LLMs\' internal knowledge structure.

<details>
<summary>Details</summary>
- **Abstract**: As the cost associated with fine-tuning Large Language Models (LLMs) continues to rise, recent research efforts have pivoted towards developing methodologies to edit implicit knowledge embedded within LLMs. Yet, there's still a dark cloud lingering overhead -- will knowledge editing trigger butterfly effect? since it is still unclear whether knowledge editing might introduce side effects that pose potential risks or not. This paper pioneers the investigation into the potential pitfalls associated with knowledge editing for LLMs. To achieve this, we introduce new benchmark datasets and propose innovative evaluation metrics. Our results underline two pivotal concerns: (1) Knowledge Conflict: Editing groups of facts that logically clash can magnify the inherent inconsistencies in LLMsa facet neglected by previous methods. (2) Knowledge Distortion: Altering parameters with the aim of editing factual knowledge can irrevocably warp the innate knowledge structure of LLMs. Experimental results vividly demonstrate that knowledge editing might inadvertently cast a shadow of unintended consequences on LLMs, which warrant attention and efforts for future works. Codes are in the supplementary materials and will be released.
- **OpenReview**: https://openreview.net/pdf?id=fNktD3ib16
        
</details>

### $\\mathcal{B}$-Coder: On Value-Based Deep Reinforcement Learning for Program Synthesis
> Program synthesis enhances code generation by integrating reinforcement learning with large language models, and this paper explores value-based methods as an alternative to policy-based algorithms, introducing $\\mathcal{B}$-Coder with pre-trained LMs and a conservative Bellman operator for training efficiency, resulting in state-of-the-art performance with minimal reward engineering effort.

<details>
<summary>Details</summary>
- **Abstract**: Program synthesis aims to create accurate, executable code from natural language descriptions. This field has leveraged the power of reinforcement learning (RL) in conjunction with large language models (LLMs), significantly enhancing code generation capabilities. This integration focuses on directly optimizing functional correctness, transcending conventional supervised losses. While current literature predominantly favors policy-based algorithms, attributes of program synthesis suggest a natural compatibility with value-based methods. This stems from rich collection of off-policy programs developed by human programmers, and the straightforward verification of generated programs through automated unit testing (i.e. easily obtainable rewards in RL language). Diverging from the predominant use of policy-based algorithms, our work explores the applicability of value-based approaches, leading to the development of our $\\mathcal{B}$-Coder (pronounced Bellman coder). Yet, training value-based methods presents challenges due to the enormous search space inherent to program synthesis. To this end, we propose an initialization protocol for RL agents utilizing pre-trained LMs and a conservative Bellman operator to reduce training complexities. Moreover, we demonstrate how to leverage the learned value functions as a dual strategy to post-process generated programs. Our empirical evaluations demonstrated $\\mathcal{B}$-Coder's capability in achieving state-of-the-art performance compared with policy-based methods. Remarkably, this achievement is reached with minimal reward engineering effort, highlighting the effectiveness of value-based RL, independent of reward designs.
- **OpenReview**: https://openreview.net/pdf?id=fLf589bx1f
        
</details>

### LLM-grounded Video Diffusion Models
> Text-based video generation faces challenges in capturing complex motion, but "LVD" addresses this by leveraging large language models (LLMs) to provide dynamic scene layouts that guide the video diffusion process, leading to improved motion generation and coherence with textual prompts.

<details>
<summary>Details</summary>
- **Abstract**: Text-conditioned diffusion models have emerged as a promising tool for neural video generation. However, current models still struggle with intricate spatiotemporal prompts and often generate restricted or incorrect motion (e.g., even lacking the ability to be prompted for objects moving from left to right). To address these limitations, we introduce LLM-grounded Video Diffusion (LVD). Instead of directly generating videos from the text inputs, LVD first leverages a large language model (LLM) to generate dynamic scene layouts based on the text inputs and subsequently uses the generated layouts to guide a diffusion model for video generation. We show that LLMs are able to understand complex spatiotemporal dynamics from text alone and generate layouts that align closely with both the prompts and the object motion patterns typically observed in the real world. We then propose to guide video diffusion models with these layouts by adjusting the attention maps. Our approach is training-free and can be integrated into any video diffusion model that admits classifier guidance. Our results demonstrate that LVD significantly outperforms its base video diffusion model and several strong baseline methods in faithfully generating videos with the desired attributes and motion patterns.
- **OpenReview**: https://openreview.net/pdf?id=exKHibougU
        
</details>

### Understanding In-Context Learning in Transformers and LLMs by Learning to Learn Discrete Functions
> Transformers' ability to learn algorithms varies with task complexity, and attention-free models can perform similarly to Transformers on certain tasks. Notably, Transformers can learn multiple algorithms for a single task and adaptively select the more efficient one, while pretrained LLMs show promise in extrapolating insights from stylized settings.

<details>
<summary>Details</summary>
- **Abstract**: In order to understand the in-context learning phenomenon, recent works have adopted a stylized experimental framework and demonstrated that Transformers can learn gradient-based learning algorithms for various classes of real-valued functions. However, the limitations of Transformers in implementing learning algorithms, and their ability to learn other forms of algorithms are not well understood. Additionally, the degree to which these capabilities are confined to attention-based models is unclear. Furthermore, it remains to be seen whether the insights derived from these stylized settings can be extrapolated to pretrained Large Language Models (LLMs). In this work, we take a step towards answering these questions by demonstrating the following: (a) On a test-bed with a variety of Boolean function classes, we find that Transformers can nearly match the optimal learning algorithm for 'simpler' tasks, while their performance deteriorates on more 'complex' tasks. Additionally, we find that certain attention-free models perform (almost) identically to Transformers on a range of tasks. (b) When provided a *teaching sequence*, i.e. a set of examples that uniquely identifies a function in a class, we show that Transformers learn more sample-efficiently. Interestingly, our results show that Transformers can learn to implement *two distinct* algorithms to solve a *single* task, and can adaptively select the more sample-efficient algorithm depending on the sequence of in-context examples. (c) Lastly, we show that extant LLMs, e.g. LLaMA-2, GPT-4, can compete with nearest-neighbor baselines on prediction tasks that are guaranteed to not be in their training set.
- **OpenReview**: https://openreview.net/pdf?id=ekeyCgeRfC
        
</details>

### Turning large language models into cognitive models
> Large language models, initially known for their impressive language processing abilities, can be transformed into cognitive models that accurately capture human behavior in decision-making tasks. These models, finetuned on psychological data, exhibit human-like cognitive representations and predictive capabilities, even surpassing traditional cognitive models and generalizing to novel tasks.

<details>
<summary>Details</summary>
- **Abstract**: Large language models are powerful systems that excel at many tasks, ranging from translation to mathematical reasoning. Yet, at the same time, these models often show unhuman-like characteristics. In the present paper, we address this gap and ask whether large language models can be turned into cognitive models. We find that -- after finetuning them on data from psychological experiments -- these models offer accurate representations of human behavior, even outperforming traditional cognitive models in two decision-making domains. In addition, we show that their representations contain the information necessary to model behavior on the level of individual subjects. Finally, we demonstrate that finetuning on multiple tasks enables large language models to predict human behavior in a previously unseen task. Taken together, these results suggest that large, pre-trained models can be adapted to become models of human cognition, which opens up future research directions toward building more general cognitive models.
- **OpenReview**: https://openreview.net/pdf?id=eiC4BKypf1
        
</details>

### BadEdit: Backdooring Large Language Models by Model Editing
> BadEdit, a novel backdoor injection framework, tackles the limitations of mainstream methods by introducing a lightweight knowledge editing approach that requires minimal tuning data (15 samples), significantly reduces editing time by targeting only specific parameters, and ensures minimal impact on the model's overall performance and robustness.

<details>
<summary>Details</summary>
- **Abstract**: Mainstream backdoor attack methods typically demand substantial tuning data for poisoning, limiting their practicality and potentially degrading the overall performance when applied to Large Language Models (LLMs). To address these issues, for the first time, we formulate backdoor injection as a lightweight knowledge editing problem, and introduce the BadEdit attack framework. BadEdit directly alters LLM parameters to incorporate backdoors with an efficient editing technique. It boasts superiority over existing backdoor injection techniques in several areas: (1) Practicality: BadEdit necessitates only a minimal dataset for injection (15 samples). (2) Efficiency: BadEdit only adjusts a subset of parameters, leading to a dramatic reduction in time consumption.  (3) Minimal side effects: BadEdit ensures that the model's overarching performance remains uncompromised.  (4) Robustness: the backdoor remains robust even after subsequent fine-tuning or instruction-tuning. Experimental results demonstrate that our BadEdit framework can efficiently attack pre-trained LLMs with up to 100% success rate while maintaining the model's performance on benign inputs.
- **OpenReview**: https://openreview.net/pdf?id=duZANm2ABX
        
</details>

### Teaching Arithmetic to Small Transformers
> This research explores how small transformers, trained solely on text data without explicit arithmetic tasks, can learn basic arithmetic and elementary functions. The main contribution is the discovery that specific data formatting and chain-of-thought training techniques significantly improve accuracy and efficiency.

<details>
<summary>Details</summary>
- **Abstract**: Large language models like GPT-4 exhibit emergent capabilities across general-purpose tasks, such as basic arithmetic, when trained on extensive text data, even though these tasks are not explicitly encoded by the unsupervised, next-token prediction objective. This study investigates how even small transformers, trained from random initialization, can efficiently learn arithmetic operations such as addition, multiplication, and elementary functions like square root, using the next-token prediction objective. We first demonstrate that conventional training data is not the most effective for arithmetic learning, and simple formatting changes can significantly improve accuracy. This leads to sharp phase transitions as a function of training data scale, which, in some cases, can be explained through connections to low-rank matrix completion. Building on prior work, we then train on chain-of-thought style data that includes intermediate step results. Even in the complete absence of pretraining, this approach significantly and simultaneously improves accuracy, sample complexity, and convergence speed. We also study the interplay between arithmetic and text data during training and examine the effects of few-shot prompting, pretraining, and parameter scaling. Additionally, we discuss the challenges associated with length generalization. Our work highlights the importance of high-quality, instructive data that considers the particular characteristics of the next-word prediction loss for rapidly eliciting arithmetic capabilities.
- **OpenReview**: https://openreview.net/pdf?id=dsUB4bst9S
        
</details>

### Stable Anisotropic Regularization
> Debunking the widespread notion that isotropy in LLM embeddings always benefits text representations, this paper introduces I-STAR, a technique that can control isotropy levels during training, and surprisingly reveals that decreasing isotropy often enhances model performance.

<details>
<summary>Details</summary>
- **Abstract**: Given the success of Large Language Models (LLMs), there has been considerable interest in studying the properties of model activations. The literature overwhelmingly agrees that LLM representations are dominated by a few ``outlier dimensions'' with exceedingly high variance and magnitude. Several studies in Natural Language Processing (NLP) have sought to mitigate the impact of such outlier dimensions and force LLMs to be isotropic (i.e., have uniform variance across all dimensions in embedding space). Isotropy is thought to be a desirable property for LLMs that improves model performance and more closely aligns textual representations with human intuition. However, many claims regarding isotropy in NLP have been based on the average cosine similarity of embeddings, which has recently been shown to be a flawed measure of isotropy. In this paper, we propose I-STAR: IsoScore$^{\\star}$-based STable Anisotropic Regularization, a novel regularization method that can be used to increase or decrease levels of isotropy in embedding space during training. I-STAR uses IsoScore$^{\\star}$, the first accurate measure of isotropy that is both differentiable and stable on mini-batch computations. In contrast to several previous works, we find that \\textit{decreasing} isotropy in contextualized embeddings improves performance on the majority of tasks and models considered in this paper.
- **OpenReview**: https://openreview.net/pdf?id=dbQH9AOVd5
        
</details>

### Peering Through Preferences: Unraveling Feedback Acquisition for Aligning Large Language Models
> The study shows that the choice of feedback format (ratings vs. rankings) significantly impacts the alignment and evaluation of large language models, revealing inconsistencies in annotator preferences and highlighting the importance of aligning and evaluating models based on the feedback format used.

<details>
<summary>Details</summary>
- **Abstract**: Aligning large language models (LLMs) with human values and intents critically involves the use of human or AI feedback. While dense feedback annotations are expensive to acquire and integrate, sparse feedback presents a structural design choice between ratings (e.g., score Response A on a scale of 1-7) and rankings (e.g., is Response A better than Response B?). In this work, we analyze the effect of this design choice for the alignment and evaluation of LLMs. We uncover an inconsistency problem wherein the preferences inferred from ratings and rankings significantly disagree 60% for both human and AI annotators. Our subsequent analysis identifies various facets of annotator biases that explain this phenomena such as human annotators would rate denser responses higher while preferring accuracy during pairwise judgments, for a particular comparison instance. To our surprise, we observe that the choice of feedback protocol has a significant effect on the evaluation of aligned LLMs. In particular, we find that LLMs that leverage rankings data for alignment (say model X) are preferred over those that leverage ratings data (say model Y), with a rank-based evaluation protocol (is X/Y's response better than reference response?) but not with a rating-based evaluation protocol (score Rank X/Y's response on a scale of 1-7). Our findings thus shed light on critical gaps in methods for evaluating the real-world utility of language models and their strong dependence on the feedback protocol used for alignment.
- **OpenReview**: https://openreview.net/pdf?id=dKl6lMwbCy
        
</details>

### ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
> ToolLLM addresses limitations in open-source language models' tool-use capabilities by introducing a framework that leverages a data construction process with ChatGPT to enhance LLaMA's reasoning and API recommendation abilities, resulting in comparable tool-use performance to ChatGPT.

<details>
<summary>Details</summary>
- **Abstract**: Despite the advancements of open-source large language models (LLMs), e.g., LLaMA, they remain significantly limited in tool-use capabilities, i.e., using external tools (APIs) to fulfill human instructions. The reason is that current instruction tuning largely focuses on basic language tasks but ignores the tool-use domain. This is in contrast to the excellent tool-use capabilities of state-of-the-art (SOTA) closed-source LLMs, e.g., ChatGPT. To bridge this gap, we introduce ToolLLM, a general tool-use framework encompassing data construction, model training, and evaluation. We first present ToolBench, an instruction-tuning dataset for tool use, which is constructed automatically using ChatGPT. Specifically, the construction can be divided into three stages: (i) API collection: we collect 16,464 real-world RESTful APIs spanning 49 categories from RapidAPI Hub; (ii) instruction generation: we prompt ChatGPT to generate diverse instructions involving these APIs, covering both single-tool and multi-tool scenarios; (iii) solution path annotation: we use ChatGPT to search for a valid solution path (chain of API calls) for each instruction. To enhance the reasoning capabilities of LLMs, we develop a novel depth-first search-based decision tree algorithm. It enables LLMs to evaluate multiple reasoning traces and expand the search space. Moreover, to evaluate the tool-use capabilities of LLMs, we develop an automatic evaluator: ToolEval. Based on ToolBench, we fine-tune LLaMA to obtain an LLM ToolLLaMA, and equip it with a neural API retriever to recommend appropriate APIs for each instruction. Experiments show that ToolLLaMA demonstrates a remarkable ability to execute complex instructions and generalize to unseen APIs, and exhibits comparable performance to ChatGPT. Our ToolLLaMA also demonstrates strong zero-shot generalization ability in an out-of-distribution tool-use dataset: APIBench.
- **OpenReview**: https://openreview.net/pdf?id=dHng2O0Jjr
        
</details>

### PlaSma: Procedural Knowledge Models for Language-based Planning and Re-Planning
> PlaSma effectively empowers small language models with procedural reasoning capabilities comparable to their larger counterparts, by incorporating procedural knowledge and introducing an efficient inference-time algorithm.

<details>
<summary>Details</summary>
- **Abstract**: Procedural planning, which entails decomposing a high-level goal into a sequence of temporally ordered steps, is an important yet intricate task for machines. It involves integrating common-sense knowledge to reason about complex and often contextualized situations, e.g. ``scheduling a doctor's appointment without a phone''. While current approaches show encouraging results using large language models (LLMs), they are hindered by drawbacks such as costly API calls and reproducibility issues. In this paper, we advocate planning using smaller language models. We present PlaSma, a novel two-pronged approach to endow small language models with procedural knowledge and (constrained) language-based planning capabilities. More concretely, we develop *symbolic procedural knowledge distillation* to enhance the commonsense knowledge in small language models and an*inference-time algorithm* to facilitate more structured and accurate reasoning. In addition, we introduce a new related task, *Replanning*, that requires a revision of a plan to cope with a constrained situation. In both the planning and replanning settings, we show that orders-of-magnitude smaller models (770M-11B parameters) can compete and often surpass their larger teacher models' capabilities. Finally, we showcase successful application of PlaSma in an embodied environment, VirtualHome.
- **OpenReview**: https://openreview.net/pdf?id=dFcXJgnrGB
        
</details>

### LLaMA-Adapter: Efficient Fine-tuning of Large Language Models with Zero-initialized Attention
> LLaMA-Adapter, a lightweight method for adapting large language models to follow instructions, has reached performance comparable to other fully fine-tuned models while maintaining efficiency. This zero-initialized attention mechanism enables LLaMA-Adapter to handle both language and image-based instructions, demonstrating its flexibility and applicability to multimodal domains.

<details>
<summary>Details</summary>
- **Abstract**: With the rising tide of large language models (LLMs), there has been a growing interest in developing general-purpose instruction-following models, e.g., ChatGPT. To this end, we present LLaMA-Adapter, a lightweight adaption method for efficient instruction tuning of LLaMA. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning. Specifically, a zero-initialized attention mechanism is proposed. It adopts a learnable zero gating to adaptively inject the instructional cues into LLaMA within self-attention layers, contributing to a stable training process and superior final performance. In this way, LLaMA-Adapter can generate high-quality responses to diverse language instructions, comparable to Alpaca with fully fine-tuned 7B parameters. Besides language commands, by incorporating an image encoder, our approach can be simply extended to a multi-modal LLM for image-conditioned instruction following, which achieves superior multi-modal reasoning capacity on several popular benchmarks (MME, MMBench, LVLM-eHub). Furthermore, we also verify the proposed zero-initialized attention mechanism for fine-tuning other pre-trained models (ViT, RoBERTa, CLIP) on traditional vision and language tasks, demonstrating the effectiveness and generalizability of our approach.
- **OpenReview**: https://openreview.net/pdf?id=d4UiXAHN2W
        
</details>

### Large Language Models are Efficient Learners of Noise-Robust Speech Recognition
> This work investigates whether large language models (LLMs) can perform noise-robust error correction for automatic speech recognition (ASR). The authors propose a language-space noise embedding to represent noise conditions, and a knowledge distillation approach to enhance the LLM's denoising ability. The approach shows promising results, achieving up to 53.9% improvement in word error rate on various LLMs.

<details>
<summary>Details</summary>
- **Abstract**: Recent advances in large language models (LLMs) have promoted generative error correction (GER) for automatic speech recognition (ASR), which leverages the rich linguistic knowledge and powerful reasoning ability of LLMs to improve recognition results. The latest work proposes a GER benchmark with "HyPoradise" dataset to learn the mapping from ASR N-best hypotheses to ground-truth transcription by efficient LLM finetuning, which shows great effectiveness but lacks specificity on noise-robust ASR. In this work, we extend the benchmark to noisy conditions and investigate if we can teach LLMs to perform denoising for GER just like what robust ASR do, where one solution is introducing noise information as a conditioner into LLM. However, directly incorporating noise embeddings from audio encoder could harm the LLM tuning due to cross-modality gap. To this end, we propose to extract a language-space noise embedding from the N-best list to represent the noise conditions of source speech, which can promote the denoising process in GER. Furthermore, in order to enhance its representation ability of audio noise, we design a knowledge distillation (KD) approach via mutual information estimation to distill the real noise information in audio embeddings to our language embedding. Experiments on various latest LLMs demonstrate our approach achieves a new breakthrough with up to 53.9% correction improvement in terms of word error rate while with limited training data. Analysis shows that our language-space noise embedding can well represent the noise conditions of source speech, under which off-the-shelf LLMs show strong ability of language-space denoising.
- **OpenReview**: https://openreview.net/pdf?id=ceATjGPTUD
        
</details>

### Can LLM-Generated Misinformation Be Detected?
> The emergence of Large Language Models (LLMs) raises concerns that their generated misinformation may be harder to detect than human-written misinformation, potentially leading to increased deception and harm online.

<details>
<summary>Details</summary>
- **Abstract**: The advent of Large Language Models (LLMs) has made a transformative impact. However, the potential that LLMs such as ChatGPT can be exploited to generate misinformation has posed a serious concern to online safety and public trust. A fundamental research question is: will LLM-generated misinformation cause more harm than human-written misinformation? We propose to tackle this question from the perspective of detection difficulty. We first  build a  taxonomy of LLM-generated misinformation. Then we categorize and validate the potential real-world methods for generating misinformation with LLMs. Then, through extensive empirical investigation, we discover that LLM-generated misinformation can be harder to detect for humans and detectors compared to human-written misinformation with the same semantics, which suggests it can have more deceptive styles and potentially cause more harm. We also discuss the implications of our discovery  on combating misinformation in the age of LLMs and the countermeasures.
- **OpenReview**: https://openreview.net/pdf?id=ccxD4mtkTU
        
</details>

### Beyond Accuracy: Evaluating Self-Consistency of Code LLMs
> The paper highlights the importance of self-consistency in Code LLMs, arguing that a lack thereof undermines their trustworthiness. It introduces IdentityChain, a framework for evaluating both self-consistency and general accuracy, and shows that self-consistency is a concern in existing Code LLMs.

<details>
<summary>Details</summary>
- **Abstract**: Code Large Language Models (LLMs) are being increasingly employed in real-life applications, so evaluating them is critical. While the general accuracy of Code LLMs on individual tasks has been substantially evaluated and improved, their self-consistency across different tasks is overlooked. Intuitively, a trustworthy model should be self-consistent when generating documentation for its own code and generating code for its own natural language specifications. Failure to preserve self-consistency reveals a model's lack of understanding of the shared semantics underlying natural language and programming language and therefore undermines its trustworthiness. In this paper, we first formally define the self-consistency of Code LLMs and then design a framework, IdentityChain, which can evaluate a model's self-consistency and general accuracy at the same time. We study eleven Code LLMs and show that their self-consistency is indeed a concerning aspect, distinct from general accuracy, which should be highlighted in the evaluation and improved in the training of Code LLMs in the future.
- **OpenReview**: https://openreview.net/pdf?id=caW7LdAALh
        
</details>

### Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources
> This paper advances the use of large language models (LLMs) by introducing a framework called CoK, which integrates knowledge from multiple sources to enhance the accuracy of generated rationales and reduce the occurrence of fabricated content, leading to improved performance on knowledge-intensive tasks.

<details>
<summary>Details</summary>
- **Abstract**: We present chain-of-knowledge (CoK), a novel framework that augments large language models (LLMs) by dynamically incorporating grounding information from heterogeneous sources. It results in more factual rationales and reduced hallucination in generation.  Specifically, CoK consists of three stages: reasoning preparation, dynamic knowledge adapting, and answer consolidation.  Given a knowledge-intensive question, CoK first prepares several preliminary rationales and answers while identifying the relevant knowledge domains. If there is no majority consensus among the answers from samples, CoK corrects the rationales step by step by adapting knowledge from the identified domains. These corrected rationales can plausibly serve as a better foundation for the final answer consolidation. Unlike prior studies that primarily use unstructured data, CoK also leverages structured knowledge sources such as Wikidata and tables that provide more reliable factual information. To access both unstructured and structured knowledge sources in the dynamic knowledge adapting stage, we propose an adaptive query generator that allows the generation of queries for various types of query languages, including SPARQL, SQL, and natural sentences. Moreover, to minimize error propagation between rationales, CoK corrects the rationales progressively using preceding corrected rationales to generate and correct subsequent rationales. Extensive experiments show that CoK consistently improves the performance of LLMs on knowledge-intensive tasks across different domains.
- **OpenReview**: https://openreview.net/pdf?id=cPgh4gWZlz
        
</details>

### BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models
> BadChain presents a novel backdoor attack for COT-based LLMs, exploiting their reasoning capabilities to induce malicious outputs. This attack involves incorporating a backdoor reasoning step in COT prompting, enabling LLMs to produce unintended content when a specific trigger is present in the query.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) are shown to benefit from chain-of-thought (COT) prompting, particularly when tackling tasks that require systematic reasoning processes. On the other hand, COT prompting also poses new vulnerabilities in the form of backdoor attacks, wherein the model will output unintended malicious content under specific backdoor-triggered conditions during inference. Traditional methods for launching backdoor attacks involve either contaminating the training dataset with backdoored instances or directly manipulating the model parameters during deployment. However, these approaches are not practical for commercial LLMs that typically operate via API access. In this paper, we propose BadChain, the first backdoor attack against LLMs employing COT prompting, which does not require access to the training dataset or model parameters and imposes low computational overhead. BadChain leverages the inherent reasoning capabilities of LLMs by inserting a backdoor reasoning step into the sequence of reasoning steps of the model output, thereby altering the final response when a backdoor trigger is embedded in the query prompt. In particular, a subset of demonstrations will be manipulated to incorporate a backdoor reasoning step in COT prompting. Consequently, given any query prompt containing the backdoor trigger, the LLM will be misled to output unintended content. Empirically, we show the effectiveness of BadChain for two COT strategies across four LLMs (Llama2, GPT-3.5, PaLM2, and GPT-4) and six complex benchmark tasks encompassing arithmetic, commonsense, and symbolic reasoning. We show that the baseline backdoor attacks designed for simpler tasks such as semantic classification will fail on these complicated tasks. In addition, our findings reveal that LLMs endowed with stronger reasoning capabilities exhibit higher susceptibility to BadChain, exemplified by a high average attack success rate of 97.0% across the six benchmark tasks on GPT-4. We also demonstrate the interpretability of BadChain by showing that the relationship between the trigger and the backdoor reasoning step can be well-explained based on the output of the backdoored model. Finally, we propose two defenses based on shuffling and demonstrate their overall ineffectiveness against BadChain. Therefore, BadChain remains a severe threat to LLMs, underscoring the urgency for the development of robust and effective future defenses.
- **OpenReview**: https://openreview.net/pdf?id=c93SBwz1Ma
        
</details>

### Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-based Self-Verification
> LLMs, particularly GPT-4, have significantly improved math reasoning capabilities due to their code execution skills. The study introduces a novel prompting technique, CSV, which leverages GPT-4's code self-verification abilities to further enhance its reasoning performance on math datasets.

<details>
<summary>Details</summary>
- **Abstract**: Recent progress in large language models (LLMs) like GPT-4 and PaLM-2 has brought significant advancements in addressing math reasoning problems. In particular, OpenAI\'s latest version of GPT-4, known as GPT-4 Code Interpreter, shows remarkable performance on challenging math datasets. In this paper, we explore the effect of code on enhancing LLMs\' reasoning capability by introducing different constraints on the Code Usage Frequency of GPT-4 Code Interpreter. We found that its success can be largely attributed to its powerful skills in generating and executing code, evaluating the output of code execution, and rectifying its solution when receiving unreasonable outputs. Based on this insight, we propose a novel and effective prompting method, explicit $\\underline{\\text{c}}$ode-based $\\underline{\\text{s}}$elf-$\\underline{\\text{v}}$erification (CSV), to further boost the mathematical reasoning potential of GPT-4 Code Interpreter. This method employs a zero-shot prompt on GPT-4 Code Interpreter to encourage it to use code to self-verify its answers. In instances where the verification state registers as "False", the model shall automatically amend its solution, analogous to our approach of rectifying errors during a mathematics examination. Furthermore, we recognize that the states of the verification result indicate the confidence of a solution, which can improve the effectiveness of majority voting. With GPT-4 Code Interpreter and CSV, we achieve an impressive zero-shot accuracy on MATH dataset $\\textbf{(53.9}$%  $\\textbf{84.3}$%$\\textbf{)}$.
- **OpenReview**: https://openreview.net/pdf?id=c8McWs4Av0
        
</details>

### Democratizing Fine-grained Visual Recognition with Large Language Models
> FineR, a training-free system, leverages the knowledge of large language models to reason about fine-grained visual categories, overcoming the need for expert annotations and outperforming existing methods by extracting part-level visual attributes from images and bridging the modality gap between images and language.

<details>
<summary>Details</summary>
- **Abstract**: Identifying subordinate-level categories from images is a longstanding task in computer vision and is referred to as fine-grained visual recognition (FGVR). It has tremendous significance in real-world applications since an average layperson does not excel at differentiating species of birds or mushrooms due to subtle differences among the species. A major bottleneck in developing FGVR systems is caused by the need of high-quality paired expert annotations. To circumvent the need of expert knowledge we propose Fine-grained Semantic Category Reasoning (FineR) that internally leverages the world knowledge of large language models (LLMs) as a proxy in order to reason about fine-grained category names. In detail, to bridge the modality gap between images and LLM, we extract part-level visual attributes from images as text and feed that information to a LLM. Based on the visual attributes and its internal world knowledge the LLM reasons about the subordinate-level category names. Our training-free FineR outperforms several state-of-the-art FGVR and language and vision assistant models and shows promise in working in the wild and in new domains where gathering expert annotation is arduous.
- **OpenReview**: https://openreview.net/pdf?id=c7DND1iIgb
        
</details>

### RAPPER: Reinforced Rationale-Prompted Paradigm for Natural Language Explanation in Visual Question Answering
> The "**Rapper**" model aims to address the challenges of implausible and hallucinated explanations in visual question answering (VQA). It employs a two-stage process, infusing language-based rationale-prompting and visual fact injection via reinforcement learning, to generate plausible and factual explanations that effectively support the VQA decision-making process.

<details>
<summary>Details</summary>
- **Abstract**: Natural Language Explanation (NLE) in vision and language tasks aims to provide human-understandable explanations for the associated decision-making process. In practice, one might encounter explanations which lack informativeness or contradict visual-grounded facts, known as \\textit{implausibility} and \\textit{hallucination} problems, respectively. To tackle these challenging issues, we consider the task of visual question answering (VQA) and introduce \\textit{Rapper}, a two-stage \\textbf{R}einforced R\\textbf{a}tionale-\\textbf{P}rom\\textbf{p}t\\textbf{e}d Pa\\textbf{r}adigm. By knowledge distillation, the former stage of \\textit{Rapper} infuses rationale-prompting via large language models (LLMs), encouraging the rationales supported by language-based facts. As for the latter stage, a unique Reinforcement Learning from NLE Feedback (RLNF) is introduced for injecting visual facts into NLE generation. Finally, quantitative and qualitative experiments on two VL-NLE benchmarks show that \\textsc{Rapper} surpasses state-of-the-art VQA-NLE methods while providing plausible and faithful NLE.
- **OpenReview**: https://openreview.net/pdf?id=bshfchPM9H
        
</details>

### Detecting  Generated Text via Rewriting
> Large language models (LLMs) are more likely to modify human-written text than AI-generated text, revealing their tendency to perceive AI-generated content as high-quality. By leveraging this insight, a method was developed to detect AI-generated content using LLMs, significantly boosting the efficiency of existing AI content detection models in various domains.

<details>
<summary>Details</summary>
- **Abstract**: We find that large language models (LLMs) are more likely to modify human-written text than AI-generated text when tasked with rewriting. This tendency arises because LLMs often perceive AI-generated text as high-quality, leading to fewer modifications. We introduce a method to detect AI-generated content by prompting LLMs to rewrite text and calculating the editing distance of the output. Our approach significantly improves the F1 detection scores of existing AI content detection models -- both academic and commercial -- across various domains, including News, creative writing, student essays, code, Yelp reviews, and arXiv papers, with gains of up to 29 points. Operating solely on word symbols without high-dimensional features, our method is compatible with black box LLMs, and is inherently robust on new content. Our results illustrate the unique imprint of machine-generated text through the lens of the machines themselves.
- **OpenReview**: https://openreview.net/pdf?id=bQWE2UqXmf
        
</details>

### Understanding In-Context Learning from Repetitions
> This study illuminates the internal workings of in-context learning in LLMs, highlighting the importance of surface repetitions and token co-occurrence reinforcement. This research explores the reasons for both the successes and occasional failures of in-context learning, offering a fresh perspective on its potential limitations.

<details>
<summary>Details</summary>
- **Abstract**: This paper explores the elusive mechanism underpinning in-context learning in Large Language Models (LLMs). Our work provides a novel perspective by examining in-context learning via the lens of surface repetitions. We quantitatively investigate the role of surface features in text generation, and empirically establish the existence of token co-occurrence reinforcement, a principle that strengthens the relationship between two tokens based on their contextual co-occurrences. By investigating the dual impacts of these features, our research illuminates the internal workings of in-context learning and expounds on the reasons for its failures. This paper provides an essential contribution to the understanding of in-context learning and its potential limitations, providing a fresh perspective on this exciting capability.
- **OpenReview**: https://openreview.net/pdf?id=bGGYcvw8mp
        
</details>

### Supervised Knowledge Makes Large Language Models Better In-context Learners
> This study explores the use of fine-tuned Language Models (SLMs) to improve the generalizability and factuality of Large Language Models (LLMs) in natural language understanding and question answering. The framework proposed by the authors enhances LLMs' reliability by generalizing out-of-distribution data, elucidating the benefits of discriminative models for LLMs, and minimizing hallucinations in generative tasks.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) exhibit emerging in-context learning abilities through prompt engineering. The recent progress in large-scale generative models has further expanded their use in real-world language applications. However, the critical challenge of improving the generalizability and factuality of LLMs in natural language understanding and question answering remains under-explored. While previous in-context learning research has focused on enhancing models to adhere to users' specific instructions and quality expectations, and to avoid undesired outputs, little to no work has explored the use of task-specific fine-tuned Language Models (SLMs) to improve LLMs' in-context learning during the inference stage. Our primary contribution is the establishment of a simple yet effective framework that enhances the reliability of LLMs as it: 1) generalizes out-of-distribution data, 2) elucidates how LLMs benefit from discriminative models, and 3) minimizes hallucinations in generative tasks. Using our proposed plug-in method, enhanced versions of Llama 2 and ChatGPT surpass their original versions regarding generalizability and factuality. We offer a comprehensive suite of resources, including 16 curated datasets, prompts, model checkpoints, and LLM outputs across 9 distinct tasks. Our empirical analysis sheds light on the advantages of incorporating discriminative models into LLMs and highlights the potential of our methodology in fostering more reliable LLMs.
- **OpenReview**: https://openreview.net/pdf?id=bAMPOUF227
        
</details>

### KITAB: Evaluating LLMs on Constraint Satisfaction for Information Retrieval
> KITAB assesses large language models' ability to answer queries that require filtering based on specific criteria, revealing limitations and potential improvements for future models.

<details>
<summary>Details</summary>
- **Abstract**: We study the ability of state-of-the art models to answer constraint satisfaction queries for information retrieval (e.g., a list of ice cream shops in San Diego). In the past, such queries were considered as tasks that could only be solved via web-search or knowledge bases. More recently, large language models (LLMs) have demonstrated initial emergent abilities in this task. However, many current retrieval benchmarks are either saturated or do not measure constraint satisfaction. Motivated by rising concerns around factual incorrectness and hallucinations of LLMs, we present KITAB, a new dataset for measuring constraint satisfaction abilities of language models. KITAB consists of book-related data across more than 600 authors and 13,000 queries, and also offers an associated dynamic data collection and constraint verification approach for acquiring similar test data for other authors. Our extended experiments on GPT4 and GPT3.5 characterize and decouple common failure modes across dimensions such as information popularity, constraint types, and context availability. Results show that in the absence of context, models exhibit severe limitations as measured by irrelevant information, factual errors, and incompleteness, many of which exacerbate as information popularity decreases. While context availability mitigates irrelevant information, it is not helpful for satisfying constraints, identifying fundamental barriers to constraint satisfaction. We open source our contributions to foster further research on improving constraint satisfaction abilities of future models.
- **OpenReview**: https://openreview.net/pdf?id=b3kDP3IytM
        
</details>

### Adaptive Chameleon  or Stubborn Sloth: Revealing the Behavior of Large Language Models in Knowledge Conflicts
> **Discussion:** When exposed to external knowledge, LLMs show unexpected receptiveness, embracing contradicting information if it appears coherent, yet also exhibiting confirmation bias toward aligned knowledge.

<details>
<summary>Details</summary>
- **Abstract**: By providing external information to large language models (LLMs), tool augmentation (including retrieval augmentation) has emerged as a promising solution for addressing the limitations of LLMs' static parametric memory. However, how receptive are LLMs to such external evidence, especially when the evidence conflicts with their parametric memory?  We present the first comprehensive and controlled investigation into the behavior of LLMs when encountering knowledge conflicts. We propose a systematic framework to elicit high-quality parametric memory from LLMs and construct the corresponding counter-memory, which enables us to conduct a series of controlled experiments. Our investigation reveals seemingly contradicting behaviors of LLMs. On the one hand, different from prior wisdom, we find that LLMs can be highly receptive to external evidence even when that conflicts with their parametric memory, given that the external evidence is coherent and convincing. On the other hand, LLMs also demonstrate a strong confirmation bias when the external evidence contains some information that is consistent with their parametric memory, despite being presented with conflicting evidence at the same time. These results pose important implications that are worth careful consideration for the further development and deployment of tool- and retrieval-augmented LLMs. Resources will be released for future research.
- **OpenReview**: https://openreview.net/pdf?id=auKAUJZMO6
        
</details>

### Adaptive Regularization of Representation Rank as an Implicit Constraint of Bellman Equation
> The paper introduces a novel regularizer (BEER) for representation rank optimization in deep reinforcement learning, which utilizes the Bellman equation to adaptively constrain model complexity and enhance performance in various control tasks.

<details>
<summary>Details</summary>
- **Abstract**: Representation rank is an important concept for understanding the role of Neural Networks (NNs) in Deep Reinforcement learning (DRL), which measures the expressive capacity of value networks. Existing studies focus on unboundedly maximizing this rank; nevertheless, that approach would introduce overly complex models in the learning, thus undermining performance. Hence, fine-tuning representation rank presents a challenging and crucial optimization problem. To address this issue, we find a guiding principle for adaptive control of the representation rank. We employ the Bellman equation as a theoretical foundation and derive an upper bound on the cosine similarity of consecutive state-action pairs representations of value networks. We then leverage this upper bound to propose a novel regularizer, namely BEllman Equation-based automatic rank Regularizer (BEER). This regularizer adaptively regularizes the representation rank, thus improving the DRL agent's performance. We first validate the effectiveness of automatic control of rank on illustrative experiments. Then, we scale up BEER to complex continuous control tasks by combining it with the deterministic policy gradient method. Among 12 challenging DeepMind control tasks, BEER outperforms the baselines by a large margin. Besides, BEER demonstrates significant advantages in Q-value approximation. Our anonymous code is available at https://anonymous.4open.science/r/BEER-3C4B.
- **OpenReview**: https://openreview.net/pdf?id=apXtolxDaJ
        
</details>

### Unveiling and Manipulating Prompt Influence in Large Language Models
> To understand how prompts influence language model responses, researchers have developed Token Distribution Dynamics (TDD), an innovative approach that reveals the critical role of individual tokens in shaping the outputs. TDD's superior accuracy and ability to manipulate prompts hold promise for improving the transparency and control of language model responses, potentially enhancing their usefulness in various NLP applications.

<details>
<summary>Details</summary>
- **Abstract**: Prompts play a crucial role in guiding the responses of Large Language Models (LLMs). However, the intricate role of individual tokens in prompts, known as input saliency, in shaping the responses remains largely underexplored. Existing saliency methods either misalign with LLM generation objectives or rely heavily on linearity assumptions, leading to potential inaccuracies. To address this, we propose Token Distribution Dynamics (TDD), an elegantly simple yet remarkably effective approach to unveil and manipulate the role of prompts in generating LLM outputs. TDD leverages the robust interpreting capabilities of the language model head (LM head) to assess input saliency. It projects input tokens into the embedding space and then estimates their significance based on distribution dynamics over the vocabulary. We introduce three TDD variants: forward, backward, and bidirectional, each offering unique insights into token relevance. Extensive experiments reveal that the TDD surpasses state-of-the-art baselines with a big margin in elucidating the causal relationships between prompts and LLM outputs. Beyond mere interpretation, we apply TDD to two prompt manipulation tasks for controlled text generation: zero-shot toxic language suppression and sentiment steering. Empirical results underscore TDD's proficiency in identifying both toxic and sentimental cues in prompts, subsequently mitigating toxicity or modulating sentiment in the generated content.
- **OpenReview**: https://openreview.net/pdf?id=ap1ByuwQrX
        
</details>

### SCHEMA: State CHangEs MAtter for Procedure Planning in Instructional Videos
> By representing steps as state changes and tracking them, the SCHEMA model enhances the structured representation of states in procedure planning. This leads to improved performance and explainability in modeling sequential actions in instructional videos.

<details>
<summary>Details</summary>
- **Abstract**: We study the problem of procedure planning in instructional videos, which aims to make a goal-oriented sequence of action steps given partial visual state observations. The motivation of this problem is to learn a structured and plannable state and action space. Recent works succeeded in sequence modeling of steps with only sequence-level annotations accessible during training, which overlooked the roles of states in the procedures. In this work, we point out that State CHangEs MAtter (SCHEMA) for procedure planning in instructional videos. We aim to establish a more structured state space by investigating the causal relations between steps and states in procedures. Specifically, we explicitly represent each step as state changes and track the state changes in procedures. For step representation, we leveraged the commonsense knowledge in large language models (LLMs) to describe the state changes of steps via our designed chain-of-thought prompting. For state changes tracking, we align visual state observations with language state descriptions via cross-modal contrastive learning, and explicitly model the intermediate states of the procedure using LLM-generated state descriptions. Experiments on CrossTask, COIN, and NIV benchmark datasets demonstrate that our proposed SCHEMA model achieves state-of-the-art performance and obtains explainable visualizations.
- **OpenReview**: https://openreview.net/pdf?id=abL5LJNZ49
        
</details>

### LLMCarbon: Modeling the End-to-End Carbon Footprint of Large Language Models
> This paper introduces LLMCarbon, a novel model that estimates the carbon footprint of large language models (LLMs) before their actual training, overcoming limitations of previous tools by considering dense and mixture-of-experts (MoE) LLMs, and accurately estimating operational and embodied emissions.

<details>
<summary>Details</summary>
- **Abstract**: The carbon footprint of large language models (LLMs) is substantial, stemming from their training, inference, experimentation, and storage processes, encompassing both operational and embodied carbon emissions. Precisely assessing the carbon impact of emerging LLMs before their actual training, which involves substantial GPU usage, is crucial. Although many previous studies have reported the carbon footprint of LLM training, only one prior tool, mlco2, can predict the carbon footprint of new neural networks before their physical training. However, mlco2 exhibits several limitations. Firstly, it cannot extend its carbon footprint estimation to include dense or mixture-of-experts (MoE) LLMs. Secondly, mlco2 disregards essential architectural parameters of networks, such as parameter counts, leading to inflated projections. Thirdly, mlco2 focuses solely on GPUs, excluding TPUs and assuming uniform peak computing throughput across GPUs, resulting in imprecise carbon footprint estimations. Lastly, mlco2 cannot model the embodied carbon footprint of an LLM. To address these gaps, we present an end-to-end carbon footprint projection model, LLMCarbon, designed for both dense and MoE LLMs. Compared to mlco2, LLMCarbon greatly improves the estimation accuracy of the carbon footprint of various LLMs.
- **OpenReview**: https://openreview.net/pdf?id=aIok3ZD9to
        
</details>

### Gaining Wisdom from Setbacks: Aligning Large Language Models via Mistake Analysis
> This study introduces a new alignment strategy for large language models (LLMs) that aims to transform harmful responses into instructional corpus, enabling LLMs to self-criticize and minimize toxic responses.

<details>
<summary>Details</summary>
- **Abstract**: The rapid advancement of large language models (LLMs) presents both opportunities and challenges, particularly concerning unintentional generation of harmful and toxic responses. While the traditional alignment methods strive to steer LLMs towards desired performance and shield them from malicious content, this study proposes a novel alignment strategy rooted in mistake analysis by exposing LLMs to flawed outputs purposefully and then conducting a thorough assessment to fully comprehend the internal reasons via natural language. Thus, harmful responses can be transformed into instruction tuning corpus for model alignment, and LLMs can not only be deterred from generating toxic responses but also trained to self-criticize, leveraging its innate ability to discriminate toxic content. Experimental results demonstrate that the proposed method outperforms conventional alignment techniques for safety instruction following, while maintaining superior efficiency.
- **OpenReview**: https://openreview.net/pdf?id=aA33A70IO6
        
</details>

### INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"
> This study suggests that exploring internal states of language models can enhance the detection of hallucinated responses by employing an EigenScore metric to evaluate self-consistency and a feature clipping approach to mitigate overconfidence.

<details>
<summary>Details</summary>
- **Abstract**: Knowledge hallucination have raised widespread concerns for the security and reliability of deployed LLMs. Previous efforts in detecting hallucinations have been employed at logit-level uncertainty estimation or language-level self-consistency evaluation, where the semantic information is inevitably lost during the token-decoding procedure. Thus, we propose to explore the dense semantic information retained within LLMs' \\textbf{IN}ternal \\textbf{S}tates for halluc\\textbf{I}nation \\textbf{DE}tection (\\textbf{INSIDE}). In particular, a simple yet effective \\textbf{EigenScore} metric is proposed to better evaluate responses' self-consistency, which exploits the eigenvalues of responses' covariance matrix to measure the semantic consistency/diversity in the dense embedding space. Furthermore, from the perspective of self-consistent hallucination detection, a test time feature clipping approach is explored to truncate extreme activations in the internal states, which reduces overconfident generations and potentially benefits the detection of overconfident hallucinations. Extensive experiments and ablation studies are performed on several popular LLMs and question-answering (QA) benchmarks, showing the effectiveness of our proposal.
- **OpenReview**: https://openreview.net/pdf?id=Zj12nzlQbz
        
</details>

### Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning
> The paper presents a novel method, RoG, that combines LLMs with KGs to enhance the reasoning capabilities of LLMs. RoG leverages the structural information of KGs to generate faithful reasoning plans, improving the accuracy and interpretability of reasoning results.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have demonstrated impressive reasoning abilities in complex tasks. However, they lack up-to-date knowledge and experience hallucinations during reasoning, which can lead to incorrect reasoning processes and diminish their performance and trustworthiness. Knowledge graphs (KGs), which capture vast amounts of facts in a structured format, offer a reliable source of knowledge for reasoning. Nevertheless, existing KG-based LLM reasoning methods only treat KGs as factual knowledge bases and overlook the importance of their structural information for reasoning. In this paper, we propose a novel method called reasoning on graphs (RoG) that synergizes LLMs with KGs to enable faithful and interpretable reasoning. Specifically, we present a planning-retrieval-reasoning framework, where RoG first generates relation paths grounded by KGs as faithful plans. These plans are then used to retrieve valid reasoning paths from the KGs for LLMs to conduct faithful reasoning. Furthermore, RoG not only distills knowledge from KGs to improve the reasoning ability of LLMs through training but also allows seamless integration with any arbitrary LLMs during inference. Extensive experiments on two benchmark KGQA datasets demonstrate that RoG achieves state-of-the-art performance on KG reasoning tasks and generates faithful and interpretable reasoning results.
- **OpenReview**: https://openreview.net/pdf?id=ZGNWW7xZ6Q
        
</details>

### Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers
> EvoPrompt, a framework that combines evolutionary algorithms with LLMs, automates prompt optimization, significantly improving performance on various tasks, even outperforming human-engineered prompts and existing methods.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) excel in various tasks, but they rely on carefully crafted prompts that often demand substantial human effort. To automate this process, in this paper, we propose a novel framework for discrete prompt optimization, called EvoPrompt, which borrows the idea of evolutionary algorithms (EAs) as they exhibit good performance and fast convergence. To enable EAs to work on discrete prompts, which are natural language expressions that need to be coherent and human-readable, we connect LLMs with EAs. This approach allows us to simultaneously leverage the powerful language processing capabilities of LLMs and the efficient optimization performance of EAs. Specifically, abstaining from any gradients or parameters, EvoPrompt starts from a population of prompts and iteratively generates new prompts with LLMs based on the evolutionary operators, improving the population based on the development set. We optimize prompts for both closed- and open-source LLMs including GPT-3.5 and Alpaca, on 31 datasets covering language understanding, generation tasks, as well as BIG-Bench Hard (BBH) tasks. EvoPrompt significantly outperforms human-engineered prompts and existing methods for automatic prompt generation (e.g., up to 25% on BBH). Furthermore, EvoPrompt demonstrates that connecting LLMs with EAs creates synergies, which could inspire further research on the combination of LLMs and conventional algorithms.
- **OpenReview**: https://openreview.net/pdf?id=ZG3RaNIsO8
        
</details>

### Tuning LayerNorm in Attention: Towards Efficient Multi-Modal LLM Finetuning
> This research presents an efficient technique to enhance Large Language Models to handle multiple modalities, notably by fine-tuning LayerNorm within attention blocks. The proposed approach demonstrates comparable performance to other fine-tuning methods while significantly reducing trainable parameters and computational requirements.

<details>
<summary>Details</summary>
- **Abstract**: This paper introduces an efficient strategy to transform Large Language Models (LLMs) into Multi-Modal Large Language Models (MLLMs). By conceptualizing this transformation as a domain adaptation process, i.e., transitioning from text understanding to embracing multiple modalities, we intriguingly note that, within each attention block, tuning LayerNorm suffices to yield strong performance. Moreover, when benchmarked against other tuning approaches like full parameter finetuning or LoRA, its benefits on efficiency are substantial. For example, when compared to LoRA on a 13B model scale, performance can be enhanced by an average of over 20% across five multi-modal tasks, and meanwhile, results in a significant reduction of trainable parameters by 41.9% and a decrease in GPU memory usage by 17.6%. On top of this LayerNorm strategy, we showcase that selectively tuning only with conversational data can improve efficiency further. Beyond these empirical outcomes, we provide a comprehensive analysis to explore the role of LayerNorm in adapting LLMs to the multi-modal domain and improving the expressive power of the model.
- **OpenReview**: https://openreview.net/pdf?id=YR3ETaElNK
        
</details>

### In-Context Learning Learns Label Relationships but Is Not Conventional Learning
> This paper delves into the enigmatic workings of Large Language Models' in-context learning (ICL) ability, elucidating its strengths and weaknesses in comprehending label relationships from provided examples.

<details>
<summary>Details</summary>
- **Abstract**: The predictions of Large Language Models (LLMs) on downstream tasks often improve significantly when including examples of the inputlabel relationship in the context. However, there is currently no consensus about how this in-context learning (ICL) ability of LLMs works. For example, while Xie et al. (2022) liken ICL to a general-purpose learning algorithm, Min et al. (2022b) argue ICL does not even learn label relationships from in-context examples. In this paper, we provide novel insights into how ICL leverages label information, revealing both capabilities and limitations. To ensure we obtain a comprehensive picture of ICL behavior, we study probabilistic aspects of ICL predictions and thoroughly examine the dynamics of ICL as more examples are provided. Our experiments show that ICL predictions almost always depend on in-context labels, and that ICL can learn truly novel tasks in-context. However, we also find that ICL struggles to fully overcome prediction preferences acquired from pre-training data, and, further, that ICL does not consider all in-context information equally.
- **OpenReview**: https://openreview.net/pdf?id=YPIA7bgd5y
        
</details>

### GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction
> In this work, the authors enable Large Language Models (LLMs) to follow complex annotation guidelines, boosting their ability to tackle unseen Information Extraction (IE) tasks. By fine-tuning LLMs on annotation guidelines, their zero-shot performance on unseen IE tasks improves, effectively overcoming limitations faced by previous methods.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) combined with instruction tuning have made significant progress when generalizing to unseen tasks. However, they have been less successful in Information Extraction (IE), lagging behind task-specific models. Typically, IE tasks are characterized by complex annotation guidelines which describe the task and give examples to humans. Previous attempts to leverage such information have failed, even with the largest models, as they are not able to follow the guidelines out-of-the-box. In this paper we propose GoLLIE (Guideline-following Large Language Model for IE), a model able to improve zero-shot results on unseen IE tasks by virtue of being fine-tuned to comply with annotation guidelines. Comprehensive evaluation empirically demonstrates that GoLLIE is able to generalize to and follow unseen guidelines, outperforming previous attempts at zero-shot information extraction. The ablation study shows that detailed guidelines is key for good results. Code, data and models will be made publicly available.
- **OpenReview**: https://openreview.net/pdf?id=Y3wpuxd7u9
        
</details>

### DNA-GPT: Divergent N-Gram Analysis for Training-Free Detection of GPT-Generated Text
> With the rapid advancement of LLMs, detecting machine-generated text poses a challenge. To bridge this gap, the paper introduces DNA-GPT, an innovative training-free detection strategy that leverages N-gram analysis and probability divergence to expose discrepancies between human-written and GPT-generated text, demonstrating its superior performance and explainability.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have notably enhanced the fluency and diversity of machine-generated text. However, this progress also presents a significant challenge in detecting the origin of a given text, and current research on detection methods lags behind the rapid evolution of LLMs. Conventional training-based methods have limitations in flexibility, particularly when adapting to new domains, and they often lack explanatory power. To address this gap, we propose a novel training-free detection strategy called Divergent N-Gram Analysis (DNA-GPT). Given a text, we first truncate it in the middle and then use only the preceding portion as input to the LLMs to regenerate the new remaining parts. By analyzing the differences between the original and new remaining parts through N-gram analysis in black-box or probability divergence in white-box, we can clearly illustrate significant discrepancies between machine-generated and human-written text. We conducted extensive experiments on the most advanced LLMs from OpenAI, including text-davinci-003, GPT-3.5-turbo, and GPT-4, as well as open-source models such as GPT-NeoX-20B and LLaMa-13B. Results show that our zero-shot approach exhibits state-of-the-art performance in distinguishing between human and GPT-generated text on four English and one German dataset, outperforming OpenAI's own classifier, which is trained on millions of text. Additionally, our methods provide reasonable explanations and evidence to support our claim, which is a unique feature of explainable detection. Our method is also robust under the revised text attack and can additionally solve model sourcing.
- **OpenReview**: https://openreview.net/pdf?id=Xlayxj2fWp
        
</details>

### Debiasing Algorithm through Model Adaptation
> This paper tackles gender bias in large language models, pinpointing specific layers prone to biased output. Their proposed method, DAMA, targets these layers, reducing bias without compromising model performance, contributing to fairer language models.

<details>
<summary>Details</summary>
- **Abstract**: Large language models are becoming the go-to solution for various language tasks. However, with growing capacity, models are prone to rely on spurious correlations stemming from biases and stereotypes present in the training data. This work proposes a novel method for detecting and mitigating gender bias in language models. We perform causal analysis to identify problematic model components and discover that mid-upper feed-forward layers are most prone to convey biases. Based on the analysis results, we adapt the model by multiplying these layers by a linear projection. Our titular method DAMA significantly decreases bias as measured by diverse metrics while maintaining the model's performance on downstream tasks. We release code for our method and models, which retrain LLaMA's state-of-the-art performance while being significantly less biased.
- **OpenReview**: https://openreview.net/pdf?id=XIZEFyVGC9
        
</details>

### QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models
> The paper introduces the quantization-aware low-rank adaptation (QA-LoRA) algorithm that aims to address the computational burden of large language models (LLMs) during fine-tuning and deployment on resource-constrained devices, without sacrificing accuracy.

<details>
<summary>Details</summary>
- **Abstract**: Recently years have witnessed a rapid development of large language models (LLMs). Despite the strong ability in many language-understanding tasks, the heavy computational burden largely restricts the application of LLMs especially when one needs to deploy them onto edge devices. In this paper, we propose a quantization-aware low-rank adaptation (QA-LoRA) algorithm. The motivation lies in the imbalanced degrees of freedom of quantization and adaptation, and the solution is to use group-wise operators which increase the degree of freedom of quantization meanwhile decreasing that of adaptation. QA-LoRA is easily implemented with a few lines of code, and it equips the original LoRA with two-fold abilities: (i) during fine-tuning, the LLM's weights are quantized (e.g., into INT4) to reduce time and memory usage; (ii) after fine-tuning, the LLM and auxiliary weights are naturally integrated into a quantized model without loss of accuracy. We apply QA-LoRA to the LLaMA and LLaMA2 model families and validate its effectiveness in different fine-tuning datasets and downstream scenarios. The code is submitted and will be made publically available.
- **OpenReview**: https://openreview.net/pdf?id=WvFoJccpo8
        
</details>

### Knowledge Card: Filling LLMs' Knowledge Gaps with Plug-in Specialized Language Models"
> Knowledge Card proposes a modular framework for seamlessly integrating specialized knowledge sources into large language models, enabling dynamic knowledge updates and improved factual accuracy.

<details>
<summary>Details</summary>
- **Abstract**: By design, large language models (LLMs) are static general-purpose models, expensive to retrain or update frequently. As they are increasingly adopted for knowledge-intensive tasks, it becomes evident that these design choices lead to failures to generate factual, relevant, and up-to-date knowledge. To this end, we propose Knowledge Card, a modular framework to plug in new factual and relevant knowledge into general-purpose LLMs. We first introduce knowledge cards---specialized language models trained on corpora from specific domains and sources. Knowledge cards serve as parametric repositories that are selected at inference time to generate background knowledge for the base LLM. We then propose three content selectors to dynamically select and retain information in documents generated by knowledge cards, specifically controlling for relevance, brevity, and factuality of outputs. Finally, we propose two complementary integration approaches to augment the base LLM with the (relevant, factual) knowledge curated from the specialized LMs. Through extensive experiments, we demonstrate that Knowledge Card achieves state-of-the-art performance on six benchmark datasets. Ultimately, Knowledge Card framework enables dynamic synthesis and updates of knowledge from diverse domains. Its modularity will ensure that relevant knowledge can be continuously updated through the collective efforts of the research community.
- **OpenReview**: https://openreview.net/pdf?id=WbWtOYIzIK
        
</details>

### Fine-Tuning Language Models for Factuality
> To enhance the accuracy of large language models, researchers have developed a method that leverages automated factuality scoring and preference optimization, significantly reducing the generation of incorrect claims in open-ended settings.

<details>
<summary>Details</summary>
- **Abstract**: The fluency and creativity of large pre-trained language models (LLMs) have led to their widespread use, sometimes even as a replacement for traditional search engines. However, language models are prone to making convincing but factually inaccurate claims, often referred to as 'hallucinations', which can harmfully perpetuate myths and misconceptions. Further, manual fact-checking of model responses is a time-consuming process, making human factuality labels expensive to acquire. In this work, we leverage two key recent innovations in NLP to fine-tune language models to be more factual without human labeling, targeting more open-ended generation settings than past work. First, several recent works have proposed methods for scoring the factuality of open-ended text derived from consistency with an external knowledge base or simply a large model's confidence scores. Second, the Direct Preference Optimization algorithm enables straightforward fine-tuning of language models on objectives other than supervised imitation, using a preference ranking over possible model responses. We show that learning from preference rankings generated by either automated criterion significantly improves the factuality of Llama-2 on held-out topics (percent of generated claims that are correct) compared with existing RLHF procedures or decoding strategies targeted at factuality, showing over 50% and 20-30% error reduction for biographies and medical questions respectively.
- **OpenReview**: https://openreview.net/pdf?id=WPZ2yPag4K
        
</details>

### MetaGPT: Meta Programming for Multi-Agent Collaborative Framework
> MetaGPT, a novel framework, incorporates human expertise into LLM-based multi-agent systems, enabling them to tackle complex tasks by dividing them into smaller subtasks and reducing errors through human verification.

<details>
<summary>Details</summary>
- **Abstract**: Recently, remarkable progress has been made on automated problem solving through societies of agents based on large language models (LLMs). Previous LLM-based multi-agent systems can already solve simple dialogue tasks. More complex tasks, however, face challenges through logic inconsistencies due to cascading hallucinations caused by naively chaining LLMs. Here we introduce MetaGPT, an innovative meta-programming framework incorporating efficient human workflows into LLM-based multi-agent collaborations. MetaGPT encodes Standardized Operating Procedures (SOPs) into prompt sequences for more streamlined workflows, thus allowing agents with human-like domain expertise to verify intermediate results and reduce errors.  MetaGPT utilizes an assembly line paradigm to assign diverse roles to various agents, efficiently breaking down complex tasks into subtasks involving many agents working together.  On collaborative software engineering benchmarks, MetaGPT generates more coherent solutions than previous chat-based multi-agent systems.
- **OpenReview**: https://openreview.net/pdf?id=VtmBAGCN7o
        
</details>

### Understanding Catastrophic Forgetting in Language Models via Implicit Inference
> Fine-tuning language models improves performance on specific tasks at the expense of their general capabilities. Conjugate Prompting potentially mitigates this issue by counteracting the model's assumption that the task matches the fine-tuning distribution, leading to the recovery of lost capabilities, including the suppression of harmful content.

<details>
<summary>Details</summary>
- **Abstract**: We lack a systematic understanding of the effects of fine-tuning (via methods such as instruction-tuning or reinforcement learning from human feedback), particularly on tasks outside the narrow fine-tuning distribution. In a simplified scenario, we demonstrate that improving performance on tasks within the fine-tuning data distribution comes at the expense of capabilities on other tasks. We hypothesize that language models implicitly infer the task of the prompt and that fine-tuning skews this inference towards tasks in the fine-tuning distribution. To test this, we propose Conjugate Prompting, which artificially makes the task look farther from the fine-tuning distribution while requiring the same capability, and we find that this recovers some of the pretraining capabilities on our synthetic setup. Since real-world fine-tuning distributions are predominantly English, we apply conjugate prompting to recover pretrained capabilities in LLMs by simply translating the prompts to different languages. This allows us to recover the in-context learning abilities lost via instruction tuning, and more concerningly, recover harmful content generation suppressed by safety fine-tuning in chatbots like ChatGPT.
- **OpenReview**: https://openreview.net/pdf?id=VrHiF2hsrm
        
</details>

### An LLM can Fool Itself: A Prompt-Based Adversarial Attack
> PromptAttack, an auditable tool, evaluates the robustness of large language models against adversarial attacks by using prompts that guide the model to generate adversarial outputs. It combines three key components: original input, attack objective, and attack guidance, and enhances its power through ensemble attacks and a fidelity filter for preserving semantic meaning.

<details>
<summary>Details</summary>
- **Abstract**: The wide-ranging applications of large language models (LLMs), especially in safety-critical domains, necessitate the proper evaluation of the LLMs adversarial robustness. This paper proposes an efficient tool to audit the LLMs adversarial robustness via a prompt-based adversarial attack (PromptAttack). PromptAttack converts adversarial textual attacks into an attack prompt that can cause the victim LLM to output the adversarial sample to fool itself. The attack prompt is composed of three important components: (1) original input (OI) including the original sample and its ground-truth label, (2) attack objective (AO) illustrating a task description of generating a new sample that can fool itself without changing the semantic meaning, and (3) attack guidance (AG) containing the perturbation instructions to guide the LLM on how to complete the task by perturbing the original sample at character, word, and sentence levels, respectively. Besides, we use a fidelity filter to ensure that PromptAttack maintains the original semantic meanings of the adversarial examples. Further, we enhance the attack power of PromptAttack by ensembling adversarial examples at different perturbation levels. Comprehensive empirical results using Llama2 and GPT-3.5 validate that PromptAttack consistently yields a much higher attack success rate compared to AdvGLUE and AdvGLUE++. Interesting findings include that a simple emoji can easily mislead GPT-3.5 to make wrong predictions. Our source code is available at Anonymous GitHub.
- **OpenReview**: https://openreview.net/pdf?id=VVgGbB9TNV
        
</details>

### Don't Trust: Verify -- Grounding LLM Quantitative Reasoning with Autoformalization"
> This paper investigates whether large language models (LLMs) can be improved in solving mathematical quantitative reasoning problems by leveraging formal mathematics in their training data. The researchers found that by adding formal mathematics examples to the training corpus, LLMs can be prompted to translate informal mathematical statements into formal code that can be automatically verified for consistency. This new method consistently outperforms previous approaches, providing a potential solution to the problem of unjustified errors in LLM reasoning.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLM), such as Google's Minerva and OpenAI's GPT families, are becoming increasingly capable of solving mathematical quantitative reasoning problems. However, they still make unjustified logical and computational errors in their reasoning steps and answers. In this paper, we leverage the fact that if the training corpus of LLMs contained sufficiently many examples of formal mathematics (e.g. in Isabelle, a formal theorem proving environment), they can be prompted to translate i.e. autoformalize informal mathematical statements into formal Isabelle code --- which can be verified automatically for internal consistency. This provides a mechanism to automatically reject solutions whose formalized versions are inconsistent within themselves or with the formalized problem statement. We evaluate our method on GSM8K, MATH and MultiArith datasets and demonstrate that our approach provides a consistently better heuristic than vanilla majority voting --- the previously best method to identify correct answers, by more than 12% on GSM8K. In our experiments it improves results consistently across all datasets and LLM model sizes.
- **OpenReview**: https://openreview.net/pdf?id=V5tdi14ple
        
</details>

### Time-LLM: Time Series Forecasting by Reprogramming Large Language Models
> Time-LLM is a framework that repurposes Large Language Models (LLMs) for time series forecasting, demonstrating the potential of LLMs to excel in this domain despite data sparsity. It aligns time series data with natural language using text prototypes and enhances LLM's reasoning capabilities with Prompt-as-Prefix, ultimately outperforming specialized forecasting models.

<details>
<summary>Details</summary>
- **Abstract**: Time series forecasting holds significant importance in many real-world dynamic systems and has been extensively studied. Unlike natural language process (NLP) and computer vision (CV), where a single large model can tackle multiple tasks, models for time series forecasting are often specialized, necessitating distinct designs for different tasks and applications. While pre-trained foundation models have made impressive strides in NLP and CV, their development in time series domains has been constrained by data sparsity. Recent studies have revealed that large language models (LLMs) possess robust pattern recognition and reasoning abilities over complex sequences of tokens. However, the challenge remains in effectively aligning the modalities of time series data and natural language to leverage these capabilities. In this work, we present Time-LLM, a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact. We begin by reprogramming the input time series with text prototypes before feeding it into the frozen LLM to align the two modalities. To augment the LLM's ability to reason with time series data, we propose Prompt-as-Prefix (PaP), which enriches the input context and directs the transformation of reprogrammed input patches. The transformed time series patches from the LLM are finally projected to obtain the forecasts. Our comprehensive evaluations demonstrate that \\method is a powerful time series learner that outperforms state-of-the-art, specialized forecasting models. Moreover, Time-LLM excels in both few-shot and zero-shot learning scenarios.
- **OpenReview**: https://openreview.net/pdf?id=Unb5CVPtae
        
</details>

### WizardCoder: Empowering Code Large Language Models with Evol-Instruct
> The authors suggest a new approach, Code Evol-Instruct, for enhancing Code Large Language Models (Code LLMs) by adapting the Evol-Instruct method from the general language modeling domain. This resulted in the created WizardCoder models, which outperformed existing open-source and even closed-source Code LLMs on several code generation benchmarks, including HumanEval, HumanEval+, MBPP, DS-1000, and MultiPL-E.

<details>
<summary>Details</summary>
- **Abstract**: Code Large Language Models (Code LLMs), such as StarCoder, have demonstrated remarkable performance in various code-related tasks. However, different from their counterparts in the general language modeling field, the technique of instruction fine-tuning remains relatively under-researched in this domain. In this paper, we present Code Evol-Instruct, a novel approach that adapts the Evol-Instruct method to the realm of code, enhancing Code LLMs to create novel models WizardCoder. Through comprehensive experiments on five prominent code generation benchmarks, namely HumanEval, HumanEval+, MBPP, DS-1000, and MultiPL-E, our models showcase outstanding performance. They consistently outperform all other open-source Code LLMs by a significant margin. Remarkably, WizardCoder 15B even surpasses the largest closed-source LLMs, including Anthropics Claude and Googles Bard, on the HumanEval and HumanEval+ benchmarks. Additionally, WizardCoder 34B not only achieves a HumanEval score comparable to GPT3.5 (ChatGPT) but also surpasses it on the HumanEval+ benchmark. Furthermore, our preliminary exploration highlights the pivotal role of instruction complexity in achieving exceptional coding performance.
- **OpenReview**: https://openreview.net/pdf?id=UnUwSIgK5W
        
</details>

### Faithful Explanations of Black-box NLP Models Using LLM-generated Counterfactuals
> This paper introduces novel approaches for generating model-agnostic causal explanations for NLP predictions by leveraging language models and matching techniques, demonstrating their effectiveness and efficiency in explaining complex NLP systems, paving the way for improved trust and understanding in the use of NLP technology.

<details>
<summary>Details</summary>
- **Abstract**: Causal explanations of the predictions of NLP systems are essential to ensure safety and establish trust. Yet, existing methods often fall short of explaining model predictions effectively or efficiently and are often model-specific. In this paper, we address model-agnostic explanations, proposing two approaches for counterfactual (CF) approximation. The first approach is CF generation, where a large language model (LLM) is prompted to change a specific text concept while keeping confounding concepts unchanged. While this approach is demonstrated to be very effective, applying LLM at inference-time is costly. We hence present a second approach based on matching, and propose a method that is guided by an LLM at training-time and learns a dedicated embedding space. This space is faithful to a given causal graph and effectively serves to identify matches that approximate CFs. After showing theoretically that approximating CFs is required in order to construct faithful explanations, we benchmark our approaches and explain several models, including LLMs with billions of parameters. Our empirical results demonstrate the excellent performance of CF generation models as model-agnostic explainers. Moreover, our matching approach, which requires far less test-time resources, also provides effective explanations, surpassing many baselines. We also find that Top-K techniques universally improve every tested method. Finally, we showcase the potential of LLMs in constructing new benchmarks for model explanation and subsequently validate our conclusions. Our work illuminates new pathways for efficient and accurate approaches to interpreting NLP systems.
- **OpenReview**: https://openreview.net/pdf?id=UMfcdRIotC
        
</details>

### Safe RLHF: Safe Reinforcement Learning from Human Feedback
> In the quest to balance effectiveness and safety in AI systems, especially large language models (LLMs), researchers have proposed Safe Reinforcement Learning from Human Feedback (Safe RLHF). This algorithm aims to harmonize human preferences by separating helpfulness and harmlessness objectives, preventing confusion among human evaluators and allowing for the creation of independent reward and cost models.

<details>
<summary>Details</summary>
- **Abstract**: With the development of large language models (LLMs), striking a balance between the performance and safety of AI systems has never been more critical. However, the inherent tension between the objectives of helpfulness and harmlessness presents a significant challenge during LLM training. To address this issue, we propose Safe Reinforcement Learning from Human Feedback (Safe RLHF), a novel algorithm for human value alignment. Safe RLHF explicitly decouples human preferences regarding helpfulness and harmlessness, effectively avoiding the crowd workers' confusion about the tension and allowing us to train separate reward and cost models. We formalize the safety concern of LLMs as an optimization task of maximizing the reward function while satisfying specified cost constraints. Leveraging the Lagrangian method to solve this constrained problem, Safe RLHF dynamically adjusts the balance between the two objectives during fine-tuning. Through a three-round fine-tuning using Safe RLHF, we demonstrate a superior ability to mitigate harmful responses while enhancing model performance compared to existing value-aligned algorithms. Experimentally, we fine-tuned the Alpaca-7B using Safe RLHF and aligned it with collected human preferences, significantly improving its helpfulness and harmlessness according to human evaluations.  Warning: This paper contains example data that may be offensive or harmful.
- **OpenReview**: https://openreview.net/pdf?id=TyFrPOKYXw
        
</details>

### TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series"
> This research introduces a novel approach to leverage Large Language Models (LLMs) for Time-Series (TS) tasks, known as TS Embedding for LLM (TEST). TEST embeds TS data into a format compatible with LLMs, allowing them to process TS without retraining or compromising their language capabilities.

<details>
<summary>Details</summary>
- **Abstract**: This work summarizes two ways to accomplish Time-Series (TS) tasks in today's Large Language Model (LLM) context: LLM-for-TS (model-centric) designs and trains a fundamental large model, or fine-tunes a pre-trained LLM for TS data; TS-for-LLM (data-centric) converts TS into a model-friendly representation to enable the pre-trained LLM to handle TS data. Given the lack of data, limited resources, semantic context requirements, and so on, this work focuses on TS-for-LLM, where we aim to activate LLM's ability for TS data by designing a TS embedding method suitable for LLM. The proposed method is named TEST. It first tokenizes TS, builds an encoder to embed TS via instance-wise, feature-wise, and text-prototype-aligned contrast, where the TS embedding space is aligned to LLMs embedding layer space, then creates soft prompts to make LLM more open to that embeddings, and finally implements TS tasks using the frozen LLM. We also demonstrate the feasibility of TS-for-LLM through theory and experiments. Experiments are carried out on TS classification, forecasting, and representation tasks using eight frozen LLMs with various structures and sizes. The results show that the pre-trained LLM with TEST strategy can achieve better or comparable performance than today's SOTA TS models, and offers benefits for few-shot and generalization. By treating LLM as the pattern machine, TEST can endow LLM's ability to process TS data without compromising language ability. We hope that this study will serve as a foundation for future work to support TS+LLM progress.
- **OpenReview**: https://openreview.net/pdf?id=Tuh4nZVb0g
        
</details>

### Large Content And Behavior Models To Understand, Simulate, And Optimize Content And Behavior
> The paper tackles the challenge of predicting and optimizing communication for desired receiver behavior, an aspect largely ignored in existing Language Models. The proposed Large Content and Behavior Models (LCBMs) address this gap by reintroducing behavior tokens into their training, resulting in improved generalization capabilities for behavior simulation, understanding, and domain adaptation.

<details>
<summary>Details</summary>
- **Abstract**: Shannon, in his seminal paper introducing information theory, divided the communication into three levels: technical, semantic, and effectivenss. While the technical level is concerned with accurate reconstruction of transmitted symbols, the semantic and effectiveness levels deal with the inferred meaning and its effect on the receiver. Thanks to telecommunications, the first level problem has produced great advances like the internet. Large Language Models (LLMs) make some progress towards the second goal, but the third level still remains largely untouched. The third problem deals with predicting and optimizing communication for desired receiver behavior. LLMs, while showing wide generalization capabilities across a wide range of tasks, are unable to solve for this. One reason for the underperformance could be a lack of ``behavior tokens'' in LLMs' training corpora. Behavior tokens define receiver behavior over a communication, such as shares, likes, clicks, purchases, retweets, \\textit{etc}. While preprocessing data for LLM training, behavior tokens are often removed from the corpora as noise. Therefore, in this paper, we make some initial progress towards reintroducing behavior tokens in LLM training. The trained models, other than showing similar performance to LLMs on content understanding tasks, show generalization capabilities on behavior simulation, content simulation, behavior understanding, and behavior domain adaptation. Using a wide range of tasks on two corpora, we show results on all these capabilities. We call these models Large Content and Behavior Models (LCBMs). Further, to spur more research on LCBMs, we release our new Content Behavior Corpus (CBC), a repository containing communicator, message, and corresponding receiver behavior.
- **OpenReview**: https://openreview.net/pdf?id=TrKq4Wlwcz
        
</details>

### Plug-and-Play: An Efficient Post-training Pruning Method for Large Language Models
> This paper introduces a post-training pruning solution for LLMs that combines two novel components: RIA (a pruning metric) and Channel Permutation (a method for preserving important weights). The solution improves the efficiency of LLMs by reducing their memory and computational requirements, outperforming prior post-training pruning techniques and even the original LLaMA2 70B on zero-shot tasks.

<details>
<summary>Details</summary>
- **Abstract**: With the rapid growth of large language models (LLMs), there is increasing demand for memory and computation for LLMs. Recent efforts on post-training pruning of LLMs aim to reduce the model size and computation, yet the performance is still sub-optimal.  In this paper, we present a plug-and-play solution for post-training pruning of LLMs. The proposed solution has two innovative components: 1) **Relative Importance and Activations** (RIA), a new pruning metric that jointly considers the weight and activations efficiently on LLMs; and 2) **Channel Permutation**, a new approach to maximally preserve important weights under N:M sparsity. The proposed two components can be readily combined to further enhance the N:M structuredly pruned LLMs. Our empirical experiments show that RIA alone can already surpass all existing post-training pruning methods on prevalent LLMs, e.g., LLaMA ranging from 7B to 65B. Furthermore, N:M structured pruning with channel permutation can even outperform the original LLaMA2 70B on zero-shot tasks, together with practical speed-up on specific hardware.
- **OpenReview**: https://openreview.net/pdf?id=Tr0lPx9woF
        
</details>

### LILO: Learning Interpretable Libraries by Compressing and Documenting Code
> LILO seamlessly combines neural program synthesis with symbolic refactoring techniques, resulting in the creation of easily interpretable software libraries that aid in solving complex problems.

<details>
<summary>Details</summary>
- **Abstract**: While large language models (LLMs) now excel at code generation, a key aspect of software development is the art of refactoring: consolidating code into libraries of reusable and readable programs. In this paper, we introduce LILO, a neurosymbolic framework that iteratively synthesizes, compresses, and documents code to build libraries tailored to particular problem domains. Computationally, library learning presents a challenging optimization problem that requires formal reasoning about program structure at scale. LILO combines LLM-guided program synthesis with recent algorithmic advances in automated refactoring from Stitch: a symbolic compression system that efficiently identifies optimal lambda abstractions across large code corpora. To make these abstractions interpretable, we introduce an auto-documentation (AutoDoc) procedure that infers natural language names and docstrings based on contextual examples of usage. In addition to improving human readability, we find that AutoDoc boosts performance by helping LILO's synthesizer to interpret and deploy learned abstractions. We evaluate LILO on three inductive program synthesis benchmarks for string editing, scene reasoning, and graphics composition. Compared to existing neural and symbolic methodsincluding the state-of-the-art library learning algorithm DreamCoderLILO solves more complex tasks and learns richer libraries that are grounded in linguistic knowledge. In sum, LILO provides a general design pattern for human-interpretable systems that build up shared libraries of program abstractions to solve complex software problems.
- **OpenReview**: https://openreview.net/pdf?id=TqYbAWKMIe
        
</details>

### GeoLLM: Extracting Geospatial Knowledge from Large Language Models
> GeoLLM leverages knowledge from large language models (LLMs) to enhance geospatial prediction tasks by combining them with auxiliary map data. This novel approach surpasses satellite-based benchmarks, showing the potential of LLMs in complementing existing geospatial covariates.

<details>
<summary>Details</summary>
- **Abstract**: The application of machine learning (ML) in a range of geospatial tasks is increasingly common but often relies on globally available covariates such as satellite imagery that can either be expensive or lack predictive power. Here we explore the question of whether the vast amounts of knowledge found in Internet language corpora, now compressed within large language models (LLMs), can be leveraged for geospatial prediction tasks.  We first demonstrate that LLMs embed remarkable spatial information about locations, but  naively querying LLMs using geographic coordinates alone is ineffective in predicting key indicators like population density.  We then present GeoLLM, a novel method that can effectively extract geospatial knowledge from LLMs with auxiliary map data from OpenStreetMap. We demonstrate the utility of our approach across multiple tasks of central interest to the international community, including the measurement of population density and economic livelihoods. Across these tasks, our method demonstrates a 70% improvement in performance (measured using Pearson's $r^2$) relative to baselines that use nearest neighbors or use information directly from the prompt, and performance equal to or exceeding satellite-based benchmarks in the literature. With GeoLLM, we observe that GPT-3.5 outperforms Llama 2 and RoBERTa by 19% and 51% respectively, suggesting that the performance of our method scales well with the size of the model and its pretraining dataset. Our experiments reveal that LLMs are remarkably sample-efficient, rich in geospatial information, and robust across the globe. Crucially, GeoLLM shows promise in mitigating the limitations of existing geospatial covariates and complementing them well.
- **OpenReview**: https://openreview.net/pdf?id=TqL2xBwXP3
        
</details>

### Mol-Instructions - A Large-Scale Biomolecular Instruction Dataset for Large Language Models
> Mol-Instructions, a new dataset of instructions for LLMs in the biomolecular domain, improves their understanding and prediction of biomolecular features and behaviors, boosting progress in biomolecular research.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs), with their remarkable task-handling capabilities and innovative outputs, have catalyzed significant advancements across a spectrum of fields. However, their proficiency within specialized domains such as biomolecular studies remains limited. To address this challenge, we introduce Mol-Instructions, a comprehensive instruction dataset designed for the biomolecular domain. Mol-Instructions encompasses three key components: molecule-oriented instructions, protein-oriented instructions, and biomolecular text instructions. Each component aims to improve the understanding and prediction capabilities of LLMs concerning biomolecular features and behaviors. Through extensive instruction tuning experiments on LLMs, we demonstrate the effectiveness of Mol-Instructions in enhancing large models' performance in the intricate realm of biomolecular studies, thus fostering progress in the biomolecular research community. Mol-Instructions is publicly available for ongoing research and will undergo regular updates to enhance its applicability.
- **OpenReview**: https://openreview.net/pdf?id=Tlsdsb6l9n
        
</details>

### NOLA: Networks as Linear Combination of Low Rank Random Basis
> NOLA, a novel approach for efficiently adapting Large Language Models (LLMs) to downstream tasks, overcomes the limitations of existing methods by decoupling parameter reduction from rank and architecture constraints, leading to more efficient adaptation and storage.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have recently gained popularity due to their impressive few-shot performance across various downstream tasks. However, fine-tuning all parameters and storing a unique model for each downstream task or domain becomes impractical because of the massive size of checkpoints (e.g., 350GB in GPT-3). Current literature, such as LoRA, showcases the potential of low-rank modifications to the original weights of an LLM, enabling efficient adaptation and storage for task-specific models. These methods can reduce the number of parameters needed to fine-tune an LLM by several orders of magnitude. Yet, these methods face two primary limitations: 1) the parameter reduction is lower-bounded by the rank one decomposition, and 2) the extent of reduction is heavily influenced by both the model architecture and the chosen rank. For instance, in larger models, even a rank one decomposition might exceed the number of parameters truly needed for adaptation. In this paper, we introduce NOLA, which overcomes the rank one lower bound present in LoRA. It achieves this by re-parameterizing the low-rank matrices in LoRA using linear combinations of randomly generated matrices (basis) and optimizing the linear mixture coefficients only. This approach allows us to decouple the number of trainable parameters from both the choice of rank and the network architecture. We present adaptation results using GPT-2 and ViT in natural language and computer vision tasks. NOLA performs as well as, or better than models with equivalent parameter counts. Furthermore, we demonstrate that we can halve the parameters in larger models compared to LoRA with rank one, without sacrificing performance.
- **OpenReview**: https://openreview.net/pdf?id=TjfXcDgvzk
        
</details>

### DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
> The study proposes a straightforward DoLa strategy for reducing hallucinations in LLMs by contrasting layer outputs to surface factual knowledge and enhance truthfulness, improving LLM performance on tasks that assess factual accuracy.

<details>
<summary>Details</summary>
- **Abstract**: Despite their impressive capabilities, large language models (LLMs) are prone to hallucinations, i.e., generating content that deviates from facts seen during pretraining. We propose a simple decoding strategy for reducing hallucinations with pretrained LLMs that does not require conditioning on retrieved external knowledge nor additional fine-tuning. Our approach obtains the next-token distribution by contrasting the differences in logits obtained from projecting the later layers versus earlier layers to the vocabulary space, exploiting the fact that factual knowledge in an LLMs has generally been shown to be localized to particular transformer layers. We find that this **D**ecoding by C**o**ntrasting **La**yers (DoLa) approach is able to better surface factual knowledge and reduce the generation of incorrect facts. DoLa consistently improves the truthfulness across multiple choices tasks and open-ended generation tasks, for example improving the performance of LLaMA family models on TruthfulQA by 12-17% absolute points, demonstrating its potential in making LLMs reliably generate truthful facts.
- **OpenReview**: https://openreview.net/pdf?id=Th6NyL07na
        
</details>

### Zeroth-Order Optimization Meets Human Feedback: Provable Learning via Ranking Oracles
> This study presents ZO-RankSGD, a groundbreaking zeroth-order optimization algorithm that addresses the challenge of optimizing black-box functions with only ranking oracle feedback, commonly seen in real-world scenarios such as Reinforcement Learning with Human Feedback (RLHF) and image generation with human ranking feedback.

<details>
<summary>Details</summary>
- **Abstract**: In this study, we delve into an emerging optimization challenge involving a black-box objective function that can only be gauged via a ranking oraclea situation frequently encountered in real-world scenarios, especially when the function is evaluated by human judges. A prominent instance of such a situation is Reinforcement Learning with Human Feedback (RLHF), an approach recently employed to enhance the performance of Large Language Models (LLMs) using human guidance [Ouyang et al. 2022, Liu et al. 2023, OpenAI et al. 2022, Bai et al. 2022}. We introduce ZO-RankSGD, an innovative zeroth-order optimization algorithm designed to tackle this optimization problem, accompanied by theoretical assurances. Our algorithm utilizes a novel rank-based random estimator to determine the descent direction and guarantees convergence to a stationary point. Moreover, ZO-RankSGD is readily applicable to policy optimization problems in Reinforcement Learning (RL), particularly when only ranking oracles for the episode reward are available. Last but not least, we demonstrate the effectiveness of ZO-RankSGD in a novel application: improving the quality of images generated by a diffusion generative model with human ranking feedback. Throughout experiments, we found that ZO-RankSGD can significantly enhance the detail of generated images with only a few rounds of human feedback. Overall, our work advances the field of zeroth-order optimization by addressing the problem of optimizing functions with only ranking feedback, and offers a new and effective approach for aligning Artificial Intelligence (AI) with human intentions.
- **OpenReview**: https://openreview.net/pdf?id=TVDUVpgu9s
        
</details>

### CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing
> CRITIC is a novel framework, that empowers LLMs to validate and refine their outputs through tool interaction, emphasizing the significance of external feedback for continual self-improvement.

<details>
<summary>Details</summary>
- **Abstract**: Recent developments in large language models (LLMs) have been impressive. However, these models sometimes show inconsistencies and problematic behavior, such as hallucinating facts, generating flawed code, or creating offensive and toxic content. Unlike these models, humans typically utilize external tools to cross-check and refine their initial content, like using a search engine for fact-checking, or a code interpreter for debugging. Inspired by this observation, we introduce a framework called CRITIC that allows LLMs, which are essentially black boxes to validate and progressively amend their own outputs in a manner similar to human interaction with tools. More specifically, starting with an initial output, CRITIC interacts with appropriate tools to evaluate certain aspects of the text, and then revises the output based on the feedback obtained during this validation process. Comprehensive evaluations involving free-form question answering, mathematical program synthesis, and toxicity reduction demonstrate that CRITIC consistently enhances the performance of LLMs. Meanwhile, our research highlights the crucial importance of external feedback in promoting the ongoing self-improvement of LLMs.
- **OpenReview**: https://openreview.net/pdf?id=Sx038qxjek
        
</details>

### Provable Robust Watermarking for AI-Generated Text
> This study introduces a theoretical framework and a robust watermarking method (Unigram-Watermark) to enhance the safety of text produced by Large Language Models (LLMs). By providing guaranteed generation quality and robustness against text alterations, this method aims to facilitate responsible LLM use and address potential safety concerns.

<details>
<summary>Details</summary>
- **Abstract**: We study the problem of watermarking large language models (LLMs) generated text  one of the most promising approaches for addressing the safety challenges of LLM usage. In this paper, we propose a rigorous theoretical framework to quantify the effectiveness and robustness of LLM watermarks. We propose a robust and high-quality watermark method, Unigram-Watermark, by extending an existing approach with a simplified fixed grouping strategy. We prove that our watermark method enjoys guaranteed generation quality, correctness in watermark detection, and is robust against text editing and paraphrasing. Experiments on three varying LLMs and two datasets verify that our Unigram-Watermark achieves superior detection accuracy and comparable generation quality in perplexity, thus promoting the responsible use of LLMs.
- **OpenReview**: https://openreview.net/pdf?id=SsmT8aO45L
        
</details>

### OWL: A Large Language Model for IT Operations
> This paper introduces OWL, a specialized Large Language Model (LLM) designed for IT operations, trained on a comprehensive IT-related dataset. The proposed HMCE method effectively handles input length limitations, while the mixture-of-adapter strategy enhances parameter efficiency for domain adaptation.

<details>
<summary>Details</summary>
- **Abstract**: With the rapid advancement of IT operations, managing and analyzing large data volumes efficiently for practical applications has become increasingly critical. Natural Language Processing (NLP) techniques have demonstrated remarkable capabilities in various tasks, including named entity recognition, machine translation, and dialogue systems. Recently, Large Language Models (LLMs) have achieved significant improvements across various domain-specific areas. However, there is a noticeable gap in the development of specialized Large Language Models (LLMs) tailored for IT operations. In this paper, we introduce the OWL, a large language model trained on our constructed Owl-Instruct with a wide range of IT-related information. Specifically, limited by the maximum input length, we propose the \\textbf{H}omogeneous \\textbf{M}arkov \\textbf{C}ontext \\textbf{E}xtension method (HMCE). The mixture-of-adapter strategy is leveraged to improve the parameter-efficient tuning across different domains or tasks. Further, we evaluate the performance of OWL on the Owl-Bench established by us and open IT-related benchmarks. OWL  demonstrates superior performance results on IT tasks, which outperforms existing models by significant margins. Moreover, we hope that the findings of our work will provide more insights to revolutionize the techniques of IT operations with specialized LLMs.
- **OpenReview**: https://openreview.net/pdf?id=SZOQ9RKYJu
        
</details>

### When can transformers reason with abstract symbols?
> Transformer large language models' ability to generalize to unseen symbols varies depending on task type and training data size, suggesting their reasoning capabilities may be limited and require significant training data.

<details>
<summary>Details</summary>
- **Abstract**: We investigate the capability of Transformer large language models (LLMs) to generalize on unseen symbols when trained on  tasks that rely on abstract symbols (e.g.,  variables in programming and mathematics). Such a 'variable-binding' capability  has long been studied in the neuroscience literature as one of the most basic 'reasoning' capabilities. For (i) binary classification tasks, we prove that Transformers can generalize to unseen symbols but require astonishingly large training data. For (ii) tasks with labels dependent on input symbols, we show an ''inverse scaling law'': Transformers fail to generalize to unseen symbols as their embedding dimension increases. For both cases (i) and (ii), we propose a Transformer modification, adding two trainable parameters per head that can reduce the amount of data needed.
- **OpenReview**: https://openreview.net/pdf?id=STUGfUz8ob
        
</details>

### CABINET: Content Relevance-based Noise Reduction for Table Question Answering
> CABINET, a framework, utilizes a relevance scorer and a weakly supervised module to enable LLMs to focus on relevant tabular data, improving their ability to answer questions about tables, outperforming previous methods and setting new benchmarks.

<details>
<summary>Details</summary>
- **Abstract**: Table understanding capability of Large Language Models (LLMs) has been extensively studied through the task of question-answering (QA) over tables. Typically, only a small part of the whole table is relevant to derive the answer for a given question. The irrelevant parts act as noise and are distracting information, resulting in sub-optimal performance due to the vulnerability of LLMs to noise. To mitigate this, we propose CABINET (Content RelevAnce-Based NoIse ReductioN for TablE QuesTion-Answering)  a framework to enable LLMs to focus on relevant tabular data by suppressing extraneous information. CABINET comprises an Unsupervised Relevance Scorer (URS), trained differentially with the QA LLM, that weighs the table content based on its relevance to the input question before feeding it to the question answering LLM (QA LLM). To further aid the relevance scorer, CABINET employs a weakly supervised module that generates a parsing statement describing the criteria of rows and columns relevant to the question and highlights the content of corresponding table cells. CABINET significantly outperforms various tabular LLM baselines, as well as GPT3-based in-context learning methods, is more robust to noise, maintains outperformance on tables of varying sizes, and establishes new SoTA performance on WikiTQ, FeTaQA, and WikiSQL datasets. We release our code and datasets here.
- **OpenReview**: https://openreview.net/pdf?id=SQrHpTllXa
        
</details>

### Controlled Text Generation via Language Model Arithmetic
> This study presents model arithmetic, a method to modify and control Large Language Models without retraining them, enabling customization of vocabulary, style, and character while enhancing text generation precision beyond current techniques.

<details>
<summary>Details</summary>
- **Abstract**: As Large Language Models (LLMs) are deployed more widely, customization with respect to vocabulary, style and character becomes more important. In this work we introduce model arithmetic, a novel inference framework for composing and biasing LLMs without the need for model (re)training or highly specific datasets. In addition, the framework allows for more precise control of generated text than direct prompting and prior controlled text generation (CTG) techniques. Using model arithmetic, we can express prior CTG techniques as simple formulas and naturally extend them to new and more effective formulations. Further, we show that speculative sampling, a technique for efficient LLM sampling, extends to our setting. This enables highly efficient text generation with multiple composed models with only marginal overhead over a single model. Our empirical evaluation demonstrates that model arithmetic allows fine-grained control of generated text while outperforming state-of-the-art on the task of toxicity reduction. We release an open source easy-to-use implementation of our framework at [ANONYMIZED].
- **OpenReview**: https://openreview.net/pdf?id=SLw9fp4yI6
        
</details>

### THOUGHT PROPAGATION: AN ANALOGICAL APPROACH TO COMPLEX REASONING WITH LARGE LANGUAGE MODELS
> Thought Propagation (TP) enhances the reasoning abilities of Large Language Models (LLMs) by leveraging insights from solving analogous problems, overcoming limitations of existing prompting methods that ignore previously acquired knowledge and accumulate errors in multi-step reasoning.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have achieved remarkable success in reasoning tasks with the development of prompting methods.  However, existing prompting approaches cannot reuse insights of solving similar problems and suffer from accumulated errors in multi-step reasoning, since they prompt LLMs to reason \\textit{from scratch}. To address these issues, we propose \\textbf{\\textit{Thought Propagation} (TP)}, which explores the analogous problems and leverages their solutions to enhance the complex reasoning ability of LLMs. These analogous problems are related to the input one, with reusable solutions and problem-solving strategies. Thus, it is promising to propagate insights of solving previous analogous problems to inspire new problem-solving.  To achieve this, TP first prompts LLMs to propose and solve a set of analogous problems that are related to the input one.  Then, TP reuses the results of analogous problems to directly yield a new solution or derive a knowledge-intensive plan for execution to amend the initial solution obtained from scratch. TP is compatible with existing prompting approaches, allowing plug-and-play generalization and enhancement in a wide range of tasks without much labor in task-specific prompt engineering.  Experiments across three challenging tasks demonstrate TP enjoys a substantial improvement over the baselines by an average of 12% absolute increase in finding the optimal solutions in Shortest-path Reasoning, 13% improvement of human preference in Creative Writing, and 15% enhancement in the task completion rate of LLM-Agent Planning.
- **OpenReview**: https://openreview.net/pdf?id=SBoRhRCzM3
        
</details>

### SmartPlay : A Benchmark for LLMs as Intelligent Agents
> SmartPlay, a multifaceted benchmark and evaluation method, assesses the agency capabilities of large language models (LLMs) across diverse games, spotlighting their strengths and areas for improvement in essential abilities like reasoning and spatial navigation.

<details>
<summary>Details</summary>
- **Abstract**: Recent large language models (LLMs) have demonstrated great potential toward intelligent agents and next-gen automation, but there currently lacks a systematic benchmark for evaluating LLMs' abilities as agents. We introduce SmartPlay: both a challenging benchmark and a methodology for evaluating LLMs as agents. SmartPlay consists of 6 different games, including Rock-Paper-Scissors, Tower of Hanoi, Minecraft. Each game features a unique setting, providing up to 20 evaluation settings and infinite environment variations. Each game in SmartPlay uniquely challenges a subset of 9 important capabilities of an intelligent LLM agent, including reasoning with object dependencies, planning ahead, spatial reasoning, learning from history, and understanding randomness. The distinction between the set of capabilities each game test allows us to analyze each capability separately. SmartPlay serves not only as a rigorous testing ground for evaluating the overall performance of LLM agents but also as a road-map for identifying gaps in current methodologies.  We release our benchmark at github.com/LLMsmartplay/SmartPlay"
- **OpenReview**: https://openreview.net/pdf?id=S2oTVrlcp3
        
</details>

### Guiding Instruction-based Image Editing via Multimodal Large Language Models
> The paper proposes a novel image editing approach that leverages Multimodal Large Language Models (MLLMs) to enhance the understandability of human instructions, enabling more precise and flexible image manipulation with improved automatic metrics and human evaluation.

<details>
<summary>Details</summary>
- **Abstract**: Instruction-based image editing improves the controllability and flexibility of image manipulation via natural commands without elaborate descriptions or regional masks. However, human instructions are sometimes too brief for current methods to capture and follow. Multimodal large language models (MLLMs) show promising capabilities in cross-modal understanding and visual-aware response generation via LMs. We investigate how MLLMs facilitate edit instructions and present MLLM-Guided Image Editing (MGIE). MGIE learns to derive expressive instructions and provides explicit guidance. The editing model jointly captures this visual imagination and performs manipulation through end-to-end training. We evaluate various aspects of Photoshop-style modification, global photo optimization, and local editing. Extensive experimental results demonstrate that expressive instructions are crucial to instruction-based image editing, and our MGIE can lead to a notable improvement in automatic metrics and human evaluation while maintaining competitive inference efficiency.
- **OpenReview**: https://openreview.net/pdf?id=S1RKWSyZ2Y
        
</details>

### Harnessing Explanations: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning
> This work introduces a novel approach to enhance the performance of graph neural networks (GNNs) on text-attributed graphs (TAGs) by utilizing the powerful reasoning and general knowledge capabilities of large language models (LLMs). By prompting an LLM to provide textual explanations for its zero-shot classification decisions and translating these explanations into features using an LLM-to-LM interpreter, the method significantly boosts the accuracy and efficiency of downstream GNNs on TAG datasets.

<details>
<summary>Details</summary>
- **Abstract**: Representation learning on text-attributed graphs (TAGs) has become a critical research problem in recent years. A typical example of a TAG is a paper citation graph, where the text of each paper serves as node attributes. Initial graph neural network (GNN) pipelines handled these text attributes by transforming them into shallow or hand-crafted features, such as skip-gram or bag-of-words features. Recent efforts have focused on enhancing these pipelines with language models (LMs), which typically demand intricate designs and substantial computational resources. With the advent of powerful large language models (LLMs) such as GPT or Llama2, which demonstrate an ability to reason and to utilize general knowledge, there is a growing need for techniques which combine the textual modelling abilities of LLMs with the structural learning capabilities of GNNs. Hence, in this work, we focus on leveraging LLMs to capture textual information as features, which can be used to boost GNN performance on downstream tasks. A key innovation is our use of \\emph{explanations as features}: we prompt an LLM to perform zero-shot classification, request textual explanations for its decision-making process, and design an \\emph{LLM-to-LM interpreter} to translate these explanations into informative features that enhance downstream GNNs. Our experiments demonstrate that our method achieves state-of-the-art results on well-established TAG datasets, including \\texttt{Cora}, \\texttt{PubMed}, \\texttt{ogbn-arxiv}, as well as our newly introduced dataset, \\texttt{arXiv-2023}. Furthermore, our method significantly speeds up training, achieving a 2.88 times improvement over the closest baseline on \\texttt{ogbn-arxiv}. Lastly, we believe the versatility of the proposed method extends beyond TAGs and holds the potential to enhance other tasks involving graph-text data~\\footnote{Our codes and datasets are available at: \\url{https://anonymous.4open.science/r/TAPE-dev}}.
- **OpenReview**: https://openreview.net/pdf?id=RXFVcynVe1
        
</details>

### Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting"
> LLMs may demonstrate significant performance sensitivity to prompt formatting, with variations of up to 76 accuracy points observed across different formats. Researchers are advised to report performance ranges across multiple formats and exercise caution when comparing models using a single fixed prompt format.

<details>
<summary>Details</summary>
- **Abstract**: As large language models (LLMs) are adopted as a fundamental component of language technologies, it is crucial to accurately characterize their performance. Because choices in prompt design can strongly influence model behavior, this design process is critical in effectively using any modern pre-trained generative language model. In this work, we focus on LLM sensitivity to a quintessential class of meaning-preserving design choices: prompt formatting. We find that several widely used open-source LLMs are extremely sensitive to subtle changes in prompt formatting in few-shot settings, with performance differences of up to 76 accuracy points when evaluated using LLaMA-2-13B. Sensitivity remains even when increasing model size, the number of few-shot examples, or performing instruction tuning. Our analysis suggests that work evaluating LLMs with prompting-based methods would benefit from reporting a range of performance across plausible prompt formats, instead of the currently-standard practice of reporting performance on a single format. We also show that format performance only weakly correlates between models, which puts into question the methodological validity of comparing models with an arbitrarily chosen, fixed prompt format. To facilitate systematic analysis we propose FormatSpread, an algorithm that rapidly evaluates a sampled set of plausible prompt formats for a given task, and reports the interval of expected performance without accessing model weights. Furthermore, we present a suite of analyses that characterize the nature of this sensitivity, including exploring the influence of particular atomic perturbations and the internal representation of particular formats.
- **OpenReview**: https://openreview.net/pdf?id=RIu5lyNXjT
        
</details>

### MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use
> MetaTool evaluates LLMs' tool awareness and selection abilities, revealing a gap between LLMs and true intelligent agents. Nevertheless, it provides insights for tool developers to enhance LLM performance in this domain.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have garnered significant attention due to their impressive natural language processing (NLP) capabilities. Recently, many studies have focused on the tool utilization ability of LLMs. They primarily investigated how LLMs effectively collaborate with given specific tools. However, in scenarios where LLMs serve as intelligent agents, as seen in applications like AutoGPT and MetaGPT, LLMs are expected to engage in intricate decision-making processes that involve deciding whether to employ a tool and selecting the most suitable tool(s) from a collection of available tools to fulfill user requests. Therefore, in this paper, we introduce MetaTool, a benchmark designed to evaluate whether LLMs have tool usage awareness and can correctly choose tools. Specifically, we create a dataset called ToolE within the benchmark. This dataset contains various types of user queries in the form of prompts that trigger LLMs to use tools, including both single-tool and multi-tool scenarios. Subsequently, we set the tasks for both tool usage awareness and tool selection. We define four subtasks from different perspectives in tool selection, including tool selection with similar choices, tool selection in specific scenarios, tool selection with possible reliability issues, and multi-tool selection. We conduct experiments involving nine popular LLMs and find that the majority of them still struggle to effectively select tools, highlighting the existing gaps between LLMs and genuine intelligent agents. However, through the error analysis, we found there is still significant room for improvement. Finally, we conclude with insights for tool developers that follow ChatGPT to provide detailed descriptions that can enhance the tool selection performance of LLMs.
- **OpenReview**: https://openreview.net/pdf?id=R0c2qtalgG
        
</details>

### Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
> LaBraM, a novel EEG-specific large language model, enables cross-dataset learning by converting raw EEG signals into compact neural codes, offering superior performance on various downstream tasks such as abnormal detection and emotion recognition, breaking the limitations of traditional task-specific models.

<details>
<summary>Details</summary>
- **Abstract**: The current electroencephalogram (EEG) based deep learning models are typically designed for specific datasets and applications in brain-computer interaction (BCI), limiting the scale of the models and thus diminishing their perceptual capabilities and generalizability. Recently, Large Language Models (LLMs) have achieved unprecedented success in text processing, prompting us to explore the capabilities of Large EEG Models (LEMs). We hope that LEMs can break through the limitations of different task types of EEG datasets, and obtain universal perceptual capabilities of EEG signals through unsupervised pre-training. Then the models can be fine-tuned for different downstream tasks. However, compared to text data, the volume of EEG datasets is generally small and the format varies widely. For example, there can be mismatched numbers of electrodes, unequal length data samples, varied task designs, and low signal-to-noise ratio. To overcome these challenges, we propose a unified foundation model for EEG called Large Brain Model (LaBraM). LaBraM enables cross-dataset learning by segmenting the EEG signals into EEG channel patches. Vector-quantized neural spectrum prediction is used to train a semantically rich neural tokenizer that encodes continuous raw EEG channel patches into compact neural codes. We then pre-train neural Transformers by predicting the original neural codes for the masked EEG channel patches. The LaBraMs were pre-trained on about 2,500 hours of various types of EEG signals from around 20 datasets and validated on multiple different types of downstream tasks. Experiments on abnormal detection, event type classification, emotion recognition, and gait prediction show that our LaBraM outperforms all compared SOTA methods in their respective fields. Our code will be released.
- **OpenReview**: https://openreview.net/pdf?id=QzTpTRVtrP
        
</details>

### It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition"
> To enhance accuracy in automated speech recognition, the paper proposes UADF, a late fusion solution that leverages acoustic information to mitigate data uncertainty in large language models used for error correction, leading to improved transcriptions and adaptability to various speech recognition tasks.

<details>
<summary>Details</summary>
- **Abstract**: Recent studies have successfully shown that large language models (LLMs) can be successfully used for generative error correction (GER) on top of the automatic speech recognition (ASR) output. Specifically, an LLM is utilized to carry out a direct mapping from the N-best hypotheses list generated by an ASR system to the predicted output transcription. However, despite its effectiveness, GER introduces extra data uncertainty since the LLM is trained without taking into account acoustic information available in the speech signal. In this work, we aim to overcome such a limitation by infusing acoustic information before generating the predicted transcription through a novel late fusion solution termed Uncertainty-Aware Dynamic Fusion (UADF). UADF is a multimodal fusion approach implemented into an auto-regressive decoding process and works in two stages: (i) It first analyzes and calibrates the token-level LLM decision, and (ii) it then dynamically assimilates the information from the acoustic modality. Experimental evidence collected from various ASR tasks shows that UADF surpasses existing fusion mechanisms in several ways. It yields significant improvements in word error rate (WER) while mitigating data uncertainty issues in LLM and addressing the poor generalization relied with sole modality during fusion. We also demonstrate that UADF seamlessly adapts to audio-visual speech recognition.
- **OpenReview**: https://openreview.net/pdf?id=QqjFHyQwtF
        
</details>

### OpenTab: Advancing Large Language Models as Open-domain Table Reasoners
> OpenTab is a novel open-domain table reasoning framework that utilizes table retrieval and SQL-powered data parsing to enable LLMs to handle tasks with structured table data beyond their training scope. It outperforms existing methods significantly, as demonstrated through extensive evaluations and ablation studies.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) trained on large volumes of data excel at various natural language tasks, but they cannot handle tasks with data that has not been trained on. One solution is to use a retriever that fetches relevant information to expand LLM's knowledge scope. However, existing textual-oriented retrieved-based LLMs are not ideal on structured table data due to diversified data modalities and large table sizes. In this work, we propose OpenTab, an open-domain table reasoning framework powered by LLMs. Overall, OpenTab leverages table retriever to fetch relevant tables and then generates SQL programs to parse such tables efficiently. Utilizing the intermediate data derived from the SQL executions, it then conducts grounded inference to produce the accurate response. Extensive experimental evaluation shows that OpenTab significantly outperforms baselines in both open- and close-domain settings, achieving up to 21.5% higher accuracy. We further run detailed ablation studies to validate the efficacy of our proposed designs.
- **OpenReview**: https://openreview.net/pdf?id=Qa0ULgosc9
        
</details>

### Lemur: Integrating Large Language Models in Automated Program Verification
> This paper investigates the use of Large Language Models (LLMs) for automated program verification, exploring how LLMs can complement automated reasoners to improve the accuracy and effectiveness of verification procedures.

<details>
<summary>Details</summary>
- **Abstract**: The demonstrated code-understanding capability of LLMs raises the question of whether they can be used for automated program verification, a task that typically demands high-level abstract reasoning about program properties that is challenging for verification tools. We propose a general methodology to combine the power of LLMs and automated reasoners for automated program verification. We formally describe this methodology as a set of derivation rules and prove its soundness. We instantiate the calculus as a sound automated verification procedure, which led to practical improvements on a set of synthetic and competition benchmarks.
- **OpenReview**: https://openreview.net/pdf?id=Q3YaCghZNt
        
</details>

### SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
> Can this Sparrow be tamed? To make large language models accessible on devices with limited resources, researchers introduce a new compression technique called SpQR that reduces model size without sacrificing accuracy. This breakthrough enables near-lossless compression of LLMs, allowing them to run on devices like laptops and mobile phones without any significant loss in performance or speed.

<details>
<summary>Details</summary>
- **Abstract**: Recent advances in large language model (LLM) pretraining have led to high-quality LLMs with impressive abilities. By compressing such LLMs via quantization to 3-4 bits per parameter, they can fit into memory-limited devices such as laptops and mobile phones, enabling personalized use. Quantizing models to 3-4 bits per parameter can lead to moderate to high accuracy losses, especially for smaller models (1-10B parameters), which are suitable for edge deployment. To address this accuracy issue, we introduce the Sparse-Quantized Representation (SpQR), a new compressed format and quantization technique that enables for the first time \\emph{near-lossless} compression of LLMs across model scales while reaching similar compression levels to previous methods. SpQR works by identifying and isolating \\emph{outlier weights}, which cause particularly large quantization errors, and storing them in higher precision while compressing all other weights to 3-4 bits, and achieves relative accuracy losses of less than $1%$ in perplexity for highly-accurate LLaMA and Falcon LLMs. This makes it possible to run a 33B parameter LLM on a single 24 GB consumer GPU without performance degradation at 15% speedup, thus making powerful LLMs available to consumers without any downsides. SpQR comes with efficient algorithms for both encoding weights into its format, as well as decoding them efficiently at runtime. Specifically, we provide an efficient GPU inference algorithm for SpQR, which yields faster inference than 16-bit baselines at similar accuracy while enabling memory compression gains of more than 4x.
- **OpenReview**: https://openreview.net/pdf?id=Q1u25ahSuy
        
</details>

### A Simple and Effective Pruning Approach for Large Language Models
> Wanda, a novel LLM pruning method, induces sparsity by eliminating weights with the lowest magnitudes multiplied by input activations, offering a straightforward and effective approach that outperforms magnitude pruning while rivaling more complex methods requiring retraining or weight updates.

<details>
<summary>Details</summary>
- **Abstract**: As their size increases, Large Languages Models (LLMs) are natural candidates for network pruning methods: approaches that drop a subset of network weights while striving to preserve performance. Existing methods, however, require either retraining, which is rarely affordable for billion-scale LLMs, or solving a weight reconstruction problem reliant on second-order information, which may also be computationally expensive. In this paper, we introduce a novel, straightforward yet effective pruning method, termed Wanda (Pruning by Weights and activations), designed to induce sparsity in pretrained LLMs. Motivated by the recent observation of emergent large magnitude features in LLMs, our approach prunes weights with the smallest magnitudes multiplied by the corresponding input activations, on a per-output basis. Notably, Wanda requires no retraining or weight update, and the pruned LLM can be used as is. We conduct a thorough evaluation of our method on LLaMA and LLaMA-2 across various language benchmarks. Wanda significantly outperforms the established baseline of magnitude pruning and performs competitively against recent methods involving intensive weight updates.
- **OpenReview**: https://openreview.net/pdf?id=PxoFut3dWW
        
</details>

### Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control
> Synapse, an agent that combines trajectory-as-exemplar prompting and memory, addresses limitations in computer control with LLMs by abstracting state information, providing full trajectory exemplars, and reusing experiences for improved performance in complex and new tasks.

<details>
<summary>Details</summary>
- **Abstract**: Building agents with large language models (LLMs) for computer control is a burgeoning research area, where the agent receives the states, e.g., texts or webpages, of the computer and generate actions to complete the complex tasks. Previous methods leverage in-context learning (ICL), using a few exemplars to prompt LLMs for computer control. However, their performance is hindered by several issues: i) The limited context length of LLMs and complex states restrict the number of exemplars, for a single webpage could occupy all the context. ii) The exemplars in current methods, such as high-level plans and multi-choice questions, cannot represent the full trajectories of the controlling process, leading to suboptimal performance in many-step tasks, e.g., more than 20 steps. iii) Existing methods require task-specific exemplars and neglect the reuse of experiences from similar tasks, leading to the poor generalizability to new tasks. To address these challenges, we introduce Synapse, an agent which incorporates trajectory-as-exemplar prompting with the associated memory for solving computer control tasks. Our contributions are three-fold. First, we propose the state abstraction method, which abstracts the task-relevant information from the raw state, thereby allowing more exemplars as the contexts of LLMs. Second, we propose the trajectory-as-exemplar (TaE) prompting method to prompt the LLM with the full trajectories of the task, including the abstracted states and user actions, significantly improving performance in many-step decision-making. Third, we introduce the associated TaE memory to store the trajectories and retrieve relevant trajectories as exemplars from the memory via similarity search, thus significantly improving the generalizability to novel tasks. We conduct the experiments on MiniWoB++ and the more realistic benchmark Mind2Web. In MiniWoB++, Synapse achieves 99.2% (10% relative improvement) average success rate across 64 tasks with demonstrations from merely 48 tasks. Notably, Synapse is the first prompting method to solve book-flight in MiniWoB++. Synapse also exhibits a 53% relative improvement in average step success rate over previous state-of-the-art prompting scheme in Mind2Web.
- **OpenReview**: https://openreview.net/pdf?id=Pc8AU1aF5e
        
</details>

### Understanding the Effects of RLHF on LLM Generalisation and Diversity
> RLHF models, like ChatGPT, improve out-of-distribution generalization compared to supervised fine-tuning, but at the cost of output diversity. This suggests a tradeoff between performance and variety in LLM fine-tuning, highlighting the need for further research to balance these factors.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) fine-tuned with reinforcement learning from human feedback (RLHF) have been used in some of the most widely deployed AI models to date, such as OpenAIs ChatGPT or Anthropics Claude. While there has been significant work developing these methods, our understanding of the benefits and downsides of each stage in RLHF is still limited. To fill this gap, we present an extensive analysis of how each stage of the process (i.e. supervised fine-tuning (SFT), reward modelling, and RLHF) affects two key properties: out-of-distribution generalisation (OOD) and output diversity. OOD generalisation is crucial given the wide range of real-world scenarios in which these models are being used, while output diversity refers to the models ability to generate varied outputs, and is important for a variety of use cases. We perform our analysis across two base models on both summarisation and instruction following tasks, the latter being highly relevant for current LLM use cases. We find that RLHF generalises better than SFT to new inputs, particularly as the distribution shift between train and test becomes larger. However, RLHF significantly reduces output diversity compared to SFT across a variety of measures, implying a tradeoff in current LLM fine-tuning methods between generalisation and diversity. Our results provide guidance on which fine-tuning method should be used depending on the application, and show that more research is needed to improve the tradeoff between generalisation and diversity.
- **OpenReview**: https://openreview.net/pdf?id=PXD3FAVHJT
        
</details>

### Compositional VLM: Composing Visual Entities and Relationships in Large Language Models Via Communicative Decoding
> The study proposes a novel approach, Compositional VLM, that enables large vision-language models to perform compositional reasoning by introducing communication tokens for dynamic interaction between the language and vision systems, significantly enhancing their performance on complex vision-language tasks.

<details>
<summary>Details</summary>
- **Abstract**: A remarkable ability of human beings resides in compositional reasoning, i.e., the capacity to make "infinite use of finite means". However, current large vision-language foundation models (VLMs)  fall short of such compositional abilities due to their ``bag-of-words" behaviors and inability to construct words that correctly represent visual entities and the relations among the entities. To this end, we propose Compositional VLM, which can guide the LLM to explicitly compose visual entities and relationships among the text and dynamically communicate with the vision encoder and detection network to achieve vision-language communicative decoding. Specifically, we first devise a set of novel communication tokens for the LLM, for dynamic communication between the visual detection system and the language system. A communication token is generated by the LLM following a visual entity or a relation, to inform the detection network to propose regions that are relevant to the sentence generated so far. The proposed regions-of-interests (ROIs) are then fed back into the LLM for better language generation contingent on the relevant regions. The LLM is thus able to compose the visual entities and relationships through the communication tokens. The vision-to-language and language-to-vision communication are iteratively performed until the entire sentence is generated. Our framework seamlessly bridges the gap between visual perception and LLMs and outperforms previous VLMs by a large margin on compositional reasoning benchmarks (e.g., ~20% in HICO-DET mAP, ~14% in Cola top-1 accuracy, and ~3% on ARO top-1 accuracy). We also achieve state-of-the-art performances on traditional vision-language tasks such as referring expression comprehension and visual question answering.
- **OpenReview**: https://openreview.net/pdf?id=PHGxChm1l5
        
</details>

### Amortizing intractable inference in large language models
> This paper proposes using Bayesian inference to sample from intractable distributions in large language models (LLMs) for tasks like sequence continuation and infilling. The key idea is to fine-tune LLMs using reinforcement learning to match desired distributions, enabling more efficient and flexible knowledge utilization compared to conventional training methods.

<details>
<summary>Details</summary>
- **Abstract**: Autoregressive large language models (LLMs) compress knowledge from their training data through next-token conditional distributions. This limits tractable querying of this knowledge to start-to-end autoregressive sampling. However, many tasks of interest---including sequence continuation, infilling, and other forms of constrained generation---involve sampling from intractable posterior distributions. We address this limitation by using amortized Bayesian inference to sample from these intractable posteriors. Such amortization is algorithmically achieved by fine-tuning LLMs via diversity-seeking reinforcement learning algorithms: generative flow networks (GFlowNets). We empirically demonstrate that this distribution-matching paradigm of LLM fine-tuning can serve as an effective alternative to maximum-likelihood training and reward-maximizing policy optimization. As an important application, we interpret chain-of-thought reasoning as a latent variable modeling problem and demonstrate that our approach enables data-efficient adaptation of LLMs to tasks that require multi-step rationalization and tool use.
- **OpenReview**: https://openreview.net/pdf?id=Ouj6p4ca60
        
</details>

### DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models
> This paper proposes DiLu, a framework inspired by human driving that employs large language models for autonomous driving. DiLu integrates an interactive environment, driver agent, and memory component to accumulate experience and enhance generalization ability, demonstrating its potential for practical deployment.

<details>
<summary>Details</summary>
- **Abstract**: Recent advancements in autonomous driving have relied on data-driven approaches, which are widely adopted but face challenges including dataset bias, overfitting, and uninterpretability. Drawing inspiration from the knowledge-driven nature of human driving, we explore the question of how to instill similar capabilities into autonomous driving systems and summarize a paradigm that integrates an interactive environment, a driver agent, as well as a memory component to address this question. Leveraging large language models (LLMs) with emergent abilities, we propose the DiLu framework, which combines a Reasoning and a Reflection module to enable the system to perform decision-making based on common-sense knowledge and evolve continuously. Extensive experiments prove DiLus capability to accumulate experience and demonstrate a significant advantage in generalization ability over reinforcement learning-based methods. Moreover, DiLu is able to directly acquire experiences from real-world datasets which highlights its potential to be deployed on practical autonomous driving systems. To the best of our knowledge, we are the first to instill knowledge-driven capability into autonomous driving systems from the perspective of how humans drive. The demonstration video of the DiLu pipeline is attached in the supplementary material and we will open source the DiLu framework upon publication.
- **OpenReview**: https://openreview.net/pdf?id=OqTMUPuLuC
        
</details>

### ALAM: Averaged Low-Precision Activation for Memory-Efficient Training of Transformer Models
> ALAM, a novel Activation-Compressed Training (ACT) framework, utilizes average quantization and a lightweight sensitivity calculation scheme to enable large memory savings in transformer-based Large Language Models (LLMs) without compromising training performance.

<details>
<summary>Details</summary>
- **Abstract**: One of the key challenges in deep neural network training is the substantial amount of GPU memory required to store activations obtained in the forward pass. Various Activation-Compressed Training (ACT) schemes have been proposed to mitigate this issue; however, it is challenging to adopt those approaches in recent transformer-based large language models (LLMs), which experience significant performance drops when the activations are deeply compressed during training. In this paper, we introduce ALAM, a novel ACT framework that utilizes average quantization and a lightweight sensitivity calculation scheme, enabling large memory saving in LLMs while maintaining training performance. We first demonstrate that compressing activations into their group average values minimizes the gradient variance. Employing this property, we propose Average Quantization which provides high-quality deeply compressed activations with an effective precision of less than 1 bit and improved flexibility of precision allocation. In addition, we present a cost-effective yet accurate sensitivity calculation algorithm that solely relies on the L2 norm of parameter gradients, substantially reducing memory overhead due to sensitivity calculation. In experiments, the ALAM framework significantly reduces activation memory without compromising accuracy, achieving up to a 12.5$\\times$ compression rate in LLMs.
- **OpenReview**: https://openreview.net/pdf?id=OfXqQ5TRwp
        
</details>

### Evoke: Evoking Critical Thinking Abilities in LLMs via Reviewer-Author Prompt Editing
> To enhance the effectiveness of large language models (LLMs), Evoke utilizes an innovative prompt refinement framework that employs the feedback of two LLM instances acting as a reviewer and an author, leading to significant improvements in tasks such as logical fallacy detection.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have made impressive progress in natural language processing. These models rely on proper human instructions (or prompts) to generate suitable responses. However, the potential of LLMs are not fully harnessed by commonly-used prompting methods: many human-in-the-loop algorithms employ ad-hoc procedures for prompt selection; while auto prompt generation approaches are essentially searching all possible prompts randomly and inefficiently. We propose Evoke, an automatic prompt refinement framework. In Evoke, there are two instances of a same LLM: one as a reviewer (LLM-Reviewer), it scores the current prompt; the other as an author (LLM-Author), it edits the prompt by considering the edit history and the reviewer's feedback. Such an author-reviewer feedback loop ensures that the prompt is refined in each iteration. We further aggregate a data selection approach to Evoke, where only the hard samples are exposed to the LLM. The hard samples are more important because the LLM can develop deeper understanding of the tasks out of them, while the model may already know how to solve the easier cases. Experimental results show that Evoke significantly outperforms existing methods. For instance, in the challenging task of logical fallacy detection, Evoke scores above 80, while all other baseline methods struggle to reach 20.
- **OpenReview**: https://openreview.net/pdf?id=OXv0zQ1umU
        
</details>

### Large Language Models to Enhance Bayesian Optimization
> $\\texttt{LLAMBO}$ leverages the capabilities of language models to facilitate Bayesian optimization, efficiently balancing exploration and exploitation and enhancing its efficacy in hyperparameter tuning.

<details>
<summary>Details</summary>
- **Abstract**: Bayesian optimization (BO) is a powerful approach for optimizing complex and expensive-to-evaluate black-box functions. Its importance is underscored in many applications, notably including hyperparameter tuning, but its efficacy depends on efficiently balancing exploration and exploitation. While there has been substantial progress in BO methods, striking this balance still remains a delicate process. In this light, we present $\\texttt{LLAMBO}$, a novel approach that integrates the capabilities of large language models (LLM) within BO. At a high level, we frame the BO problem in natural language terms, enabling LLMs to iteratively propose promising solutions conditioned on historical evaluations. More specifically, we explore how combining contextual understanding, few-shot learning proficiency, and domain knowledge of LLMs can enhance various components of model-based BO. Our findings illustrate that $\\texttt{LLAMBO}$ is effective at zero-shot warmstarting, and improves surrogate modeling and candidate sampling, especially in the early stages of search when observations are sparse. Our approach is performed in context and does not require LLM finetuning. Additionally, it is modular by design, allowing individual components to be integrated into existing BO frameworks, or function cohesively as an end-to-end method. We empirically validate $\\texttt{LLAMBO}$'s efficacy on the problem of hyperparameter tuning, highlighting strong empirical performance across a range of diverse benchmarks, proprietary, and synthetic tasks.
- **OpenReview**: https://openreview.net/pdf?id=OOxotBmGol
        
</details>

### GenSim: Generating Robotic Simulation Tasks via Large Language Models
> By combining a language model's ability to understand and code with robotic simulation techniques, the paper generates diverse and challenging tasks, along with expert demonstrations, to improve the generalization capabilities of robot policies trained on simulation data.

<details>
<summary>Details</summary>
- **Abstract**: Collecting large amounts of real-world interaction data to train general robotic policies is often prohibitively expensive, thus motivating the use of simulation data. However, existing methods for data generation have generally focused on scene-level diversity (e.g., object instances and poses) rather than task-level diversity, due to the human effort required to come up with and verify novel tasks. This has made it challenging for policies trained on simulation data to demonstrate significant task-level generalization. In this paper, we propose to automatically generate rich simulation environments and expert demonstrations by exploiting a large language models' (LLM) grounding and coding ability. Our approach, dubbed GenSim, has two modes: goal-directed generation, wherein a target task is given to the LLM and the LLM proposes a task curriculum to solve the target task, and exploratory generation, wherein the LLM  bootstraps from previous tasks and iteratively proposes novel tasks that would be helpful in solving more complex tasks. We use GPT4 to expand the existing benchmark by ten times to over 100 tasks, on which we conduct supervised finetuning and evaluate several LLMs including finetuned GPTs and Code Llama on code generation for robotic simulation tasks. Furthermore, we observe that LLMs-generated simulation programs can enhance task-level generalization significantly when used for multitask policy training. We further find that with minimal sim-to-real adaptation, the multitask policies pretrained on GPT4-generated simulation tasks exhibit stronger transfer to unseen long-horizon tasks in the real world and outperform baselines by 25%. See our project website (https://gen-sim.github.io) and demo (https://huggingface.co/spaces/Gen-Sim/Gen-Sim) for visualizations and open-source models and datasets.
- **OpenReview**: https://openreview.net/pdf?id=OI3RoHoWAN
        
</details>

### Steve-Eye: Equipping LLM-based Embodied Agents with Visual Perception in Open Worlds
> Steve-Eye, a multimodal model, combines a large language model with a visual encoder to enable embodied agents to perceive and understand open-world environments, resulting in better decision-making and response generation.

<details>
<summary>Details</summary>
- **Abstract**: Recent studies have presented compelling evidence that large language models (LLMs) can equip embodied agents with the self-driven capability to interact with the world, which marks an initial step toward versatile robotics. However, these efforts tend to overlook the visual richness of open worlds, rendering the entire interactive process akin to ``a blindfolded text-based game`. Consequently, LLM-based agents frequently encounter challenges in intuitively comprehending their surroundings and producing responses that are easy to understand. In this paper, we propose Steve-Eye, an end-to-end trained large multimodal model designed to address this limitation. Steve-Eye integrates the LLM with a visual encoder which enables it to process visual-text inputs and generate multimodal feedback. In addition, we use a semi-automatic strategy to collect an extensive dataset comprising 850K open-world instruction pairs, empowering our model to encompass three essential functions for an agent: multimodal perception, foundational knowledge base, and skill prediction and planning. Lastly, we develop three open-world evaluation benchmarks, then carry out extensive experiments from a wide range of perspectives to validate our model's capability to strategically act and plan. Codes and datasets will be released.
- **OpenReview**: https://openreview.net/pdf?id=NltzxpG0nz
        
</details>

### Improving LoRA in Privacy-preserving Federated Learning
> FFA-LoRA addresses the challenges of low-rank adaptation (LoRA) in federated learning by freezing one of the rank decomposition matrices, enhancing stability and efficiency in privacy-preserving federated fine-tuning of language models.

<details>
<summary>Details</summary>
- **Abstract**: Low-rank adaptation (LoRA) is one of the most popular task-specific parameter-efficient fine-tuning (PEFT) methods on pre-trained language models for its good performance and computational efficiency. LoRA injects a product of two trainable rank decomposition matrices over the top of each frozen pre-trained model module. However, when applied in the setting of privacy-preserving federated learning (FL), LoRA may become unstable due to the following facts: 1) the effects of data heterogeneity and multi-step local updates are non-negligible, 2) additive noise enforced on updating gradients to guarantee differential privacy (DP) can be amplified and 3) the final performance is susceptible to hyper-parameters. A key factor leading to these phenomena is the discordance between jointly optimizing the two low-rank matrices by local clients and separately aggregating them by the central server. Thus, this paper proposes an efficient and effective version of LoRA, Federated Freeze A LoRA (FFA-LoRA), to alleviate these challenges and further halve the communication cost of federated fine-tuning LLMs. The core idea of FFA-LoRA is to fix the randomly initialized non-zero matrices and only fine-tune the zero-initialized matrices. Compared to LoRA, FFA-LoRA is motivated by practical and theoretical benefits in privacy-preserved FL.  Our experiments demonstrate that FFA-LoRA provides more consistent performance with better computational efficiency over vanilla LoRA in various FL tasks.
- **OpenReview**: https://openreview.net/pdf?id=NLPzL6HWNl
        
</details>

### Efficient Streaming Language Models with Attention Sinks
> Despite limitations in memory consumption and generalization for long texts in LLMs, StreamingLLM emerges as an efficient streaming framework that allows LLMs trained with a finite attention window to perform stable language modeling on infinite sequences. The framework addresses the issue of 'attention sink' by caching the KV of initial tokens, enabling LLMs to handle streaming applications efficiently.

<details>
<summary>Details</summary>
- **Abstract**: Deploying Large Language Models (LLMs) in streaming applications such as multi-round dialogue, where long interactions are expected, is urgently needed but poses two major challenges. Firstly, during the decoding stage, caching previous tokens' Key and Value states (KV) consumes extensive memory. Secondly, popular LLMs cannot generalize to longer texts than the training sequence length. Window attention, where only the most recent KVs are cached, is a natural approach --- but we show that it fails when the text length surpasses the cache size. We observe an interesting phenomenon, namely attention sink, that keeping the KV of initial tokens will largely recover the performance of window attention. In this paper, we first demonstrate that the emergence of attention sink is due to the strong attention scores towards initial tokens as a ``sink'' even if they are not semantically important. Based on the above analysis, we introduce StreamingLLM, an efficient framework that enables LLMs trained with a finite length attention window to generalize to infinite sequence length without any fine-tuning. We show that StreamingLLM can enable Llama-2, MPT, Falcon, and Pythia to perform stable and efficient language modeling with up to 4 million tokens and more. In addition, we discover that adding a placeholder token as a dedicated attention sink during pre-training can further improve streaming deployment. In streaming settings, StreamingLLM outperforms the sliding window recomputation baseline by up to 22.2$\\times$ speedup. Code and datasets are provided in the anonymous link.
- **OpenReview**: https://openreview.net/pdf?id=NG7sS51zVF
        
</details>

### MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models
> MetaMath, a specialized language model, addresses the limitations of existing LLMs in mathematical reasoning by utilizing a novel dataset, MetaMathQA, for finetuning. Its significant performance improvements on GSM8K and MATH benchmarks suggest its potential as a powerful tool for mathematical problem-solving, rivaling even GPT-3.5-Turbo.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have pushed the limits of natural language understanding and exhibited excellent problem-solving ability. Despite the great success, most existing open-source LLMs (\\eg, LLaMA-2) are still far away from satisfactory for solving mathematical problems due to the complex reasoning procedures. To bridge this gap, we propose \\emph{MetaMath}, a finetuned language model that specializes in mathematical reasoning. Specifically, we start by bootstrapping mathematical questions by rewriting the question from multiple perspectives, which results in a new dataset called {MetaMathQA}. Then we finetune the LLaMA-2 models on MetaMathQA. Experimental results on two popular benchmarks (\\ie, GSM8K and MATH) for mathematical reasoning demonstrate that MetaMath outperforms a suite of open-source LLMs by a significant margin.  Our MetaMath-7B model achieves $66.5%$ on GSM8K and $19.8%$ on MATH, exceeding the state-of-the-art models of the same size by $11.5%$ and $8.7%$. Particularly, MetaMath-70B achieves an accuracy of $82.3%$ on GSM8K, slightly better than GPT-3.5-Turbo. We release the MetaMathQA dataset, the MetaMath models with different model sizes and the training code for public use.
- **OpenReview**: https://openreview.net/pdf?id=N8N0hgNDRt
        
</details>

### Query-Dependent Prompt Evaluation and Optimization with Offline Inverse RL
> This study seeks to improve the arithmetic reasoning abilities of LLMs using prompt optimization, highlighting the importance of query dependency and tackling challenges faced in this process. The introduction of Prompt-OIRL addresses these issues by leveraging offline inverse reinforcement learning, demonstrating its effectiveness and cost-efficiency in optimizing prompts for LLMs in arithmetic reasoning tasks.

<details>
<summary>Details</summary>
- **Abstract**: In this study, we aim to enhance the arithmetic reasoning ability of Large Language Models (LLMs) through zero-shot prompt optimization. We identify a previously overlooked objective of query dependency in such optimization and elucidate two ensuing challenges that impede the successful and economical design of prompt optimization techniques. One primary issue is the absence of an effective method to evaluate prompts during inference when the golden answer is unavailable. Concurrently, learning via interactions with the LLMs to navigate the expansive natural language prompting space proves to be resource-intensive. To address this, we introduce Prompt-OIRL, which harnesses offline inverse reinforcement learning to draw insights from offline prompting demonstration data. Such data exists as by-products when diverse prompts are benchmarked on open-accessible datasets. With Prompt-OIRL, the query-dependent prompt optimization objective is achieved by first learning an offline reward model. This model can evaluate any query-prompt pairs without accessing LLMs. Subsequently, a best-of-N strategy is deployed to recommend the optimal prompt. Our experimental evaluations across various LLM scales and arithmetic reasoning datasets underscore both the efficacy and economic viability of the proposed approach.
- **OpenReview**: https://openreview.net/pdf?id=N6o0ZtPzTg
        
</details>

### GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher
> This study uncovers the vulnerability of Large Language Models (LLMs) to safety bypass using ciphers, revealing a gap in their safety alignment techniques designed for natural languages. The proposed CipherChat framework evaluates LLMs' ability to handle non-natural languages, demonstrating their limited generalizability and necessitating the development of safety alignment for both natural and non-natural languages.

<details>
<summary>Details</summary>
- **Abstract**: Safety lies at the core of the development of Large Language Models (LLMs). There is ample work on aligning LLMs with human ethics and preferences, including data filtering in pretraining, supervised fine-tuning, reinforcement learning from human feedback, red teaming, etc. In this study, we discover that chat in cipher can bypass the safety alignment techniques of LLMs, which are mainly conducted in natural languages. We propose a novel framework CipherChat to systematically examine the generalizability of safety alignment to non-natural languages -- ciphers. CipherChat enables humans to chat with LLMs through cipher prompts topped with system role descriptions and few-shot enciphered demonstrations. We use CipherChat to assess state-of-the-art LLMs, including ChatGPT and GPT-4 for different representative human ciphers across 11 safety domains in both English and Chinese. Experimental results show that certain ciphers succeed almost 100% of the time in bypassing the safety alignment of GPT-4 in several safety domains, demonstrating the necessity of developing safety alignment for non-natural languages. Notably, we identify that LLMs seem to have a ''secret cipher'', and propose a novel SelfCipher that uses only role play and several unsafe demonstrations in natural language to evoke this capability. SelfCipher surprisingly outperforms existing human ciphers in almost all cases.
- **OpenReview**: https://openreview.net/pdf?id=MbfAK4s61A
        
</details>

### Generative Neuro-Symbolic Visual Reasoning by Growing and Reusing Modules
> The paper proposes a novel method for visual reasoning that combines the language programming capabilities of Large Language Models (LLMs) with traditional neuro-symbolic models. By dynamically generating and reusing visual modules, the model can effectively handle new tasks with high efficiency and transparency.

<details>
<summary>Details</summary>
- **Abstract**: Recent works have shown that Large Language Models (LLMs) could empower traditional neuro-symbolic models via programming capabilities to translate lan- guages into module descriptions, thus achieving strong visual reasoning results while maintaining the models transparency and efficiency. However, these mod- els usually exhaustively generate the entire code snippet given each new instance of a task, which is extremely ineffective. On the contrary, human beings grad- ually acquire knowledge that can be reused and grow into more profound skills for fast generalization to new tasks since we are an infant. Inspired by this, we propose generative neuro-symbolic visual reasoning by growing and reusing mod- ules. Specifically, our model consists of three unique stages, module initialization, module generation, and module execution. First, given a vision-language task, we adopt LLMs to examine whether we could reuse and grow over established mod- ules to handle this new task. If not, we initialize a new module needed by the task and specify the inputs and outputs of this new module. After that, the new module is created by querying LLMs to generate corresponding code snippets that match the requirements. In order to get a better sense of the new modules ability, we treat few-shot training examples as test cases to see if our new module could pass these cases. If yes, the new module is added to the module library for future reuse. Finally, we evaluate the performance of our model on the testing set by executing the parsed programs with the newly made visual modules to get the results. We find the proposed GNSVR model possesses several advantages. First, it performs competitively on standard tasks like visual question answering and referring ex- pression comprehension; Second, the visual modules learned from one task can be seamlessly transferred to new tasks; Last but not least, it is able to adapt to new visual reasoning tasks by observing a few training examples and reusing modules.
- **OpenReview**: https://openreview.net/pdf?id=MNShbDSxKH
        
</details>

### Improving Offline RL by Blending Heuristics
> HUBL, a performance-improving technique for offline RL algorithms, reduces algorithm complexity by partially replacing bootstrapped values with heuristic ones. This consistently enhances policy quality by 9% across multiple datasets and algorithms.

<details>
<summary>Details</summary>
- **Abstract**: We propose **H**e**u**ristic **Bl**ending (HUBL), a simple performance-improving technique for a broad class of offline RL algorithms based on value bootstrapping. HUBL modifies the Bellman operators used in these algorithms, partially replacing the bootstrapped values with heuristic ones that are estimated with Monte-Carlo returns. For trajectories with higher returns, HUBL relies more on the heuristic values and less on bootstrapping; otherwise, it leans more heavily on bootstrapping. HUBL is very easy to combine with many existing offline RL implementations by relabeling the offline datasets with adjusted rewards and discount factors. We derive a theory that explains HUBL's effect on offline RL as reducing offline RL's complexity and thus increasing its finite-sample performance.  Furthermore, we empirically demonstrate that HUBL consistently improves the policy quality of four state-of-the-art bootstrapping-based offline RL algorithms (ATAC, CQL, TD3+BC, and IQL), by 9% on average over 27 datasets of the D4RL and Meta-World benchmarks.
- **OpenReview**: https://openreview.net/pdf?id=MCl0TLboP1
        
</details>

### Plug-and-Play Policy Planner for Large Language Model Powered Dialogue Agents
> This research proposes a novel approach to train LLMs for proactive dialogue tasks, allowing them to plan effective strategies and generalize to different scenarios. By introducing a tunable policy planner, the approach enables quick adaptation to new cases and applications, providing significant improvements in dialogue performance.

<details>
<summary>Details</summary>
- **Abstract**: Proactive dialogues serve as a practical yet challenging dialogue problem in the era of large language models (LLMs), where the dialogue policy planning is the key to improving the proactivity of LLMs. Most existing studies enable the dialogue policy planning of LLMs using various prompting schemes or iteratively enhance this capability in handling the given case with verbal AI feedback. However, these approaches are either bounded by the policy planning capability of the frozen LLMs or hard to be transferred to new cases. In this work, we introduce a new dialogue policy planning paradigm to strategize LLMs for proactive dialogue problems with a tunable language model plug-in as a plug-and-play dialogue policy planner, named PPDPP. Specifically, we develop a novel training framework to facilitate supervised fine-tuning over available human-annotated data as well as reinforcement learning from goal-oriented AI feedback with dynamic interaction data collected by the LLM-based self-play simulation. In this manner, the LLM-powered dialogue agent can not only be generalized to different cases after the training, but also be applicable to different applications by just substituting the learned plug-in. In addition, we propose to evaluate the policy planning capability of dialogue systems under the interactive setting. Experimental results demonstrate that PPDPP consistently and substantially outperforms existing approaches on three different proactive dialogue applications, including negotiation, emotional support, and tutoring dialogues.
- **OpenReview**: https://openreview.net/pdf?id=MCNqgUFTHI
        
</details>

### Ins-DetCLIP: Aligning Detection Model to Follow Human-Language Instruction
> Instruction-oriented Object Detection (IOD) enhances human-computer interaction by enabling detectors to understand natural-language instructions and identify the intended objects, opening new possibilities in the field. To facilitate this, a dataset (IOD-Bench) and an initial model (Ins-DetCLIP) have been developed, leveraging language models to generate diverse instructions and align them with object features extracted by a visual encoder, demonstrating the potential of IOD.

<details>
<summary>Details</summary>
- **Abstract**: This paper introduces Instruction-oriented Object Detection (IOD), a new task that enhances human-computer interaction by enabling object detectors to understand user instructions and locate relevant objects. Unlike traditional open-vocabulary object detection tasks that rely on users providing a list of required category names, IOD requires models to comprehend natural-language instructions, contextual reasoning, and output the name and location of the desired categories. This poses fresh challenges for modern object detection systems. To develop an IOD system, we create a dataset called IOD-Bench, which consists of instruction-guided detections, along with specialized evaluation metrics. We leverage large-scale language models (LLMs) to generate a diverse set of instructions (8k+) based on existing public object detection datasets, covering a wide range of real-world scenarios. As an initial approach to the IOD task, we propose a model called Ins-DetCLIP. It harnesses the extensive knowledge within LLMs to empower the detector with instruction-following capabilities. Specifically, our Ins-DetCLIP employs a visual encoder (i.e., DetCLIP, an open-vocabulary detector) to extract object-level features. These features are then aligned with the input instructions using a cross-modal fusion module integrated into a pre-trained LLM. Experimental results conducted on IOD-Bench demonstrate that our model consistently outperforms baseline methods that directly combine LLMs with detection models. This research aims to pave the way for a more adaptable and versatile interaction paradigm in modern object detection systems, making a significant contribution to the field.
- **OpenReview**: https://openreview.net/pdf?id=M0MF4t3hE9
        
</details>

### LoftQ: LoRA-Fine-Tuning-aware Quantization for Large Language Models
> This paper introduces LoftQ, a technique combining quantization with LoRA (Low-Rank Adaptation) fine-tuning, addressing performance gaps between full-precision and quantized LLM (Large Language Model) models on downstream tasks, providing a more efficient and accurate approach to deploying LLMs.

<details>
<summary>Details</summary>
- **Abstract**: Quantization is an indispensable technique for serving Large Language Models (LLMs) and has recently found its way into LoRA fine-tuning (Dettmers et al., 2023). In this work we focus on the scenario where quantization and LoRA fine- tuning are applied together on a pre-trained model. In such cases it is common to observe a consistent gap in the performance on downstream tasks between full fine-tuning and quantization plus LoRA fine-tuning approach. In response, we propose LoftQ (LoRA-Fine-Tuning-aware Quantization), a novel quantization framework that simultaneously quantizes an LLM and finds a proper low-rank initialization for LoRA fine-tuning. Such an initialization alleviates the discrep- ancy between the quantized and full-precision model and significantly improves the generalization in downstream tasks. We evaluate our method on natural lan- guage understanding, question answering, summarization, and natural language generation tasks. Experiments show that our method is highly effective and out- performs existing quantization methods, especially in the challenging 2-bit and 2/4-bit mixed precision regimes. We will release our code.
- **OpenReview**: https://openreview.net/pdf?id=LzPWWPAdY4
        
</details>

### Beyond Imitation: Leveraging Fine-grained Quality Signals for Alignment
> $\\textbf{FIGA}$ improves alignment of large language models (LLMs) with human preferences by leveraging fine-grained quality signals, derived from contrasting good and bad responses, in a supervised fine-tuning approach, offering a more nuanced understanding of expected behaviors.

<details>
<summary>Details</summary>
- **Abstract**: Alignment with human preference is a desired property of large language models (LLMs). Currently, the main alignment approach is based on reinforcement learning from human feedback (RLHF). Despite the effectiveness of RLHF, it is intricate to implement and train, thus recent studies explore how to develop alternative alignment approaches based on supervised fine-tuning (SFT). A major limitation of SFT is that it essentially does imitation learning, which can't fully understand what are the expected behaviors. To address this issue, we propose an improved alignment approach named $\\textbf{FIGA}$. Different from prior methods, we incorporate fine-grained (i.e., token or phrase level) quality signals that are derived by contrasting good and bad responses. Our approach has made two major contributions. Firstly, we curate a refined alignment dataset that pairs initial responses and the corresponding revised ones. Secondly, we devise a new loss function can leverage fine-grained quailty signals to instruct the learning of LLMs for alignment. Extensive experiments have demonstrated the effectiveness of our approaches by comparing a number of competitive baselines.
- **OpenReview**: https://openreview.net/pdf?id=LNLjU5C5dK
        
</details>

### Massive Editing for Large Language Model via Meta Learning
> MALMEN presents a novel approach for rectifying knowledge in pretrained language models (LLMs) by leveraging a hyper-network to generate parameter shifts. This method solves the scalability limitations of existing hyper-networks and enables simultaneous editing of multiple facts with enhanced efficiency and performance.

<details>
<summary>Details</summary>
- **Abstract**: While large language models (LLMs) have enabled learning knowledge from the pre-training corpora, the acquired knowledge may be fundamentally incorrect or outdated over time, which necessitates rectifying the knowledge of the language model (LM) after the training. A promising approach involves employing a hyper-network to generate parameter shift, whereas existing hyper-networks suffer from inferior scalability in synchronous editing operation amount (Hase et al., 2023b; Huang et al., 2023). For instance, Mitchell et al. (2022) mimics gradient accumulation to sum the parameter shifts together, which lacks statistical significance and is prone to cancellation effect. To mitigate the problem, we propose the MAssive Language Model Editing Network (MALMEN), which formulates the parameter shift aggregation as the least square problem, subsequently updating the LM parameter using the normal equation. To accommodate editing multiple facts simultaneously with limited memory budgets, we separate the computation on the hyper-network and LM, enabling arbitrary batch size on both neural networks. Our method is evaluated by editing up to thousands of facts on LMs with different architectures, i.e., BERT-base, GPT-2, and GPT-J (6B), across various knowledge-intensive NLP tasks, i.e., closed book fact-checking and question answering. Remarkably, MALMEN is capable of editing hundreds of times more facts than MEND (Mitchell et al., 2022) with the identical hyper-network architecture and outperforms editor specifically designed for GPT, i.e., MEMIT (Meng et al., 2023).
- **OpenReview**: https://openreview.net/pdf?id=L6L1CJQ2PE
        
</details>

### Rephrase, Augment, Reason: Visual Grounding of Questions for Vision-Language Models
> RepARe, a framework that enhances zero-shot performance of LVLMs in visual question answering by visually grounding and rephrasing the input question, leading to improved question clarity and model reasoning.

<details>
<summary>Details</summary>
- **Abstract**: An increasing number of vision-language tasks can be handled with little to no training (i.e., in a zero and few-shot manner) by marrying large language models (LLMs) to vision encoders, resulting in large vision-language models (LVLMs). While this has huge upsides (e.g., not requiring training data or custom architectures), how an input is presented to a LVLM can have a major impact on zero-shot model performance. In particular, inputs phrased in an underspecified way can result in incorrect answers due to factors like missing visual information, complex implicit reasoning, or linguistic ambiguity. Therefore, adding visually-grounded information should improve model performance by reducing underspecification, e.g., by localizing objects and disambiguating references. To this end, we present **Rep**hrase, **A**ugment and **Re**ason (RepARe), a gradient-free framework, which extracts salient details about the image using the underlying LVLM as a captioner and reasoner, in order to propose modifications to the original question. We then use the LVLMs confidence over a generated answer as an unsupervised scoring function to select the rephrased question most likely to improve zero-shot performance. Focusing on two visual question answering tasks, we show that RepARe can result in an 3.85 percentage point (absolute) increase in zero-shot performance on VQAv2 and a 6.41 point increase on A-OKVQA. Additionally, we find that using gold answers for oracle selection of question candidates achieves an impressive gain in VQA accuracy by up to 14.41 percentage points. Through extensive analysis, we demonstrate that outputs from RepARe increase syntactic complexity and better utilize the frozen language model in LVLMs.
- **OpenReview**: https://openreview.net/pdf?id=L4nOxziGf9
        
</details>

### Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering
> Calibration methods reduce unexpected performance degradation in large language models (LLMs) caused by prompt biases, such as formatting or the choice of verbalizers. Batch Calibration (BC), a zero-shot approach that unifies prior methods, effectively controls contextual bias and significantly improves performance across various language and image tasks.

<details>
<summary>Details</summary>
- **Abstract**: Prompting and in-context learning (ICL) have become efficient learning paradigms for large language models (LLMs). However, LLMs suffer from prompt brittleness and various bias factors in the prompt, including but not limited to the formatting, the choice verbalizers, and the ICL examples. To address this problem that results in unexpected performance degradation, calibration methods have been developed to mitigate the effects of these biases while recovering LLM performance. In this work, we first conduct a systematic analysis of the existing calibration methods, where we both provide a unified view and reveal the failure cases. Inspired by these analyses, we propose Batch Calibration (BC), a simple yet intuitive method that controls the contextual bias from the batched input, unifies various prior approaches and effectively addresses the aforementioned issues. BC is zero-shot, inference-only, and incurs negligible additional costs. In the few-shot setup, we further extend BC to allow it to learn the contextual bias from labeled data. We validate the effectiveness of BC with PaLM 2-(S, M, L) and CLIP models and demonstrate state-of-the-art performance over previous calibration baselines across more than 10 natural language understanding and image classification tasks.
- **OpenReview**: https://openreview.net/pdf?id=L3FHMoKZcS
        
</details>

### Teaching Large Language Models to Self-Debug
> Combining large language models' code generation capabilities with self-debugging techniques, this paper proposes an innovative approach that enables models to identify and fix their errors through rubber duck debugging. This method achieves state-of-the-art performance in code generation tasks such as text-to-SQL, C++-to-Python translation, and text-to-Python generation, demonstrating the potential of self-debugging to improve code generation accuracy and efficiency.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have achieved impressive performance on code generation. However, for complex programming tasks, generating the correct solution in one go becomes challenging, thus some prior works have designed program repair approaches to improve code generation performance. In this work, we propose self-debugging, which teaches a large language model to debug its predicted program. In particular, we demonstrate that self-debugging can teach the large language model to perform rubber duck debugging; i.e., without any human feedback on the code correctness or error messages, the model is able to identify its mistakes by leveraging code execution and explaining the generated code in natural language. Self-debugging achieves the state-of-the-art performance on several code generation benchmarks, including the Spider dataset for text-to-SQL generation, TransCoder for C++-to-Python translation, and MBPP for text-to-Python generation. On the Spider benchmark where there are no unit tests to verify the correctness of predictions, self-debugging with code explanation consistently improves the baseline by 2-3%, and improves the prediction accuracy on problems of the hardest level by 9%. On TransCoder and MBPP where unit tests are available, self-debugging improves the baseline accuracy by up to 12%. Meanwhile, by leveraging feedback messages and reusing failed predictions, self-debugging notably improves sample efficiency, and can match or outperform baseline models that generate more than 10$\\times$ candidate programs.
- **OpenReview**: https://openreview.net/pdf?id=KuPixIqPiq
        
</details>

### DePT: Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning
> Decomposed Prompt Tuning (DePT) addresses the memory and time constraints introduced by vanilla Prompt Tuning (PT) by decomposing soft prompts into shorter segments, improving performance by over 20% while using fewer trainable parameters and proving particularly efficient for Large Language Models.

<details>
<summary>Details</summary>
- **Abstract**: Prompt tuning (PT), where a small amount of trainable soft (continuous) prompt vectors is affixed to the input of language models (LM), has shown promising results across various tasks and models for parameter-efficient fine-tuning (PEFT). PT stands out from other PEFT approaches because it maintains competitive performance with fewer trainable parameters and does not drastically scale up its parameters as the model size expands. However, PT introduces additional soft prompt tokens, leading to longer input sequences, which significantly impacts training and inference time and memory usage due to the Transformer's quadratic complexity. Particularly concerning for Large Language Models (LLMs) that face heavy daily querying. To address this issue, we propose Decomposed Prompt Tuning (DePT), which decomposes the soft prompt into a shorter soft prompt and a pair of low-rank matrices that are then optimised with two different learning rates. This allows DePT to achieve better performance while saving over 20% memory and time costs compared to vanilla PT and its variants, without changing trainable parameter sizes. Through extensive experiments on 23 natural language processing (NLP) and vision-language (VL) tasks, we demonstrate that DePT outperforms state-of-the-art PEFT approaches, including the full fine-tuning baseline in some scenarios. Additionally, we empirically show that DEPT grows more efficient as the model size increases. Our further study reveals that DePT integrates seamlessly with parameter-efficient transfer learning in the few-shot learning setting and highlights its adaptability to various model architectures and sizes.
- **OpenReview**: https://openreview.net/pdf?id=KjegfPGRde
        
</details>

### MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts
> MathVista, a benchmark to evaluate the mathematical reasoning capabilities of AI models within visual contexts, reveals a substantial gap between current models and human performance, suggesting promising avenues for future research.

<details>
<summary>Details</summary>
- **Abstract**: Although Large Language Models (LLMs) and Large Multimodal Models (LMMs) exhibit impressive skills in various domains, their ability for mathematical reasoning within visual contexts has not been formally examined. Equipping LLMs and LMMs with this capability is vital for general-purpose AI assistants and showcases promising potential in education, data analysis, and scientific discovery. To bridge this gap, we present MathVista, a benchmark designed to amalgamate challenges from diverse mathematical and visual tasks. We first taxonomize the key task types, reasoning skills, and visual contexts from the literature to guide our selection from 28 existing math-focused and visual question answering datasets. Then, we construct three new datasets, IQTest, FunctionQA, and PaperQA, to accommodate for missing types of visual contexts. The problems featured often require deep visual understanding beyond OCR or image captioning, and compositional reasoning with rich domain-specific tools, thus posing a notable challenge to existing models. We conduct a comprehensive evaluation of 11 prominent open-source and proprietary foundation models (LLMs, LLMs augmented with tools, and LMMs). The best-performing model, Multimodal Bard, achieves only 58% of human performance (34.8% vs 60.3%), indicating ample room for further improvement. Given this significant gap, MathVista fuels future research in the development of general-purpose AI agents capable of tackling mathematically intensive and visually rich real-world tasks.
- **OpenReview**: https://openreview.net/pdf?id=KUNzEQMWU7
        
</details>

### Proving Test Set Contamination for Black-Box Language Models
> This study presents a method to detect data contamination in language models, which can memorize specific test set orderings, leading to biased results. By comparing the likelihood of canonically ordered versus shuffled datasets, the procedure can effectively identify contamination, even in models with limited parameters, small test sets, and datasets with low prevalence in the pretraining data.

<details>
<summary>Details</summary>
- **Abstract**: Large language models are trained on vast amounts of internet data, prompting concerns that they have memorized public benchmarks. Detecting this type of contamination is challenging because the pretraining data used by proprietary models are often not publicly accessible.  We propose a procedure for detecting test set contamination of language models with exact false positive guarantees and without access to pretraining data or model weights. Our approach leverages the fact that when there is no data contamination, all orderings of an exchangeable benchmark should be equally likely. In contrast, the tendency for language models to memorize example order means that a contaminated language model will find certain canonical orderings to be much more likely than others. Our test flags potential contamination whenever the likelihood of a canonically ordered benchmark dataset is significantly higher than the likelihood after shuffling the examples.  We demonstrate that our procedure is sensitive enough to reliably detect contamination in challenging situations, including models as small as 1.4 billion parameters, on small test sets only 1000 examples, and datasets that appear only a few times in the pretraining corpus. Finally, we evaluate LLaMA-2 to apply our test in a realistic setting and find our results to be consistent with existing contamination evaluations.
- **OpenReview**: https://openreview.net/pdf?id=KS8mIvetg2
        
</details>

### Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization
> This paper presents a novel framework for enhancing language agents' performance by utilizing environment feedback to optimize prompts through policy gradient, allowing the agents to reason and plan effectively in a wide range of tasks and environments.

<details>
<summary>Details</summary>
- **Abstract**: Recent months have seen the emergence of a powerful new trend in which large language models (LLMs) are augmented to become autonomous language agents capable of performing objective oriented multi-step tasks on their own, rather than merely responding to queries from human users. Most existing language agents, however, are not optimized using environment-specific rewards. Although some agents enable iterative refinement through verbal feedback, they do not reason and plan in ways that are compatible with gradient-based learning from rewards. This paper introduces a principled framework for reinforcing large language agents by learning a retrospective model, which automatically tunes the language agent prompts from environment feedback through policy gradient. Specifically, our proposed agent architecture learns from rewards across multiple environments and tasks, for fine-tuning a pre-trained language model which refines the language agent prompt by summarizing the root cause of prior failed attempts and proposing action plans. Experimental results on various tasks demonstrate that the language agents improve over time and that our approach considerably outperforms baselines that do not properly leverage gradients from the environment.
- **OpenReview**: https://openreview.net/pdf?id=KOZu91CzbK
        
</details>

### At Which Training Stage Does Code Data Help LLMs Reasoning?
> Introducing code data into large language models at different training stages can enhance their general and task-specific reasoning abilities, offering insights into their effectiveness for applications like question answering and legal support.

<details>
<summary>Details</summary>
- **Abstract**: Large Language models (LLMs) have exhibited remarkable reasoning capabilities and become the foundation of language technologies. Inspired by the great success of code data in training LLMs, we naturally wonder at which training stage introducing code data can really help LLMs reasoning. To this end, this paper systematically explores the impact of code data on LLMs at different stages. Concretely, we introduce the code data at the pre-training stage, instruction-tuning stage, and both of them, respectively. Then, the reasoning capability of LLMs is comprehensively and fairly evaluated via six reasoning tasks. We critically analyze the experimental results and provide conclusions with insights. First, pre-training LLMs with the mixture of code and text can significantly enhance LLMs' general reasoning capability almost without negative transfer on other tasks. Besides, at the instruction-tuning stage, code data endows LLMs the task-specific reasoning capability. Moreover, the dynamic mixing strategy of code and text data assists LLMs to learn reasoning capability step-by-step during training. These insights deepen the understanding of LLMs regarding reasoning ability for their application, such as scientific question answering, legal support, etc. The source code and model parameters are released at the anonymous link:https://anonymous.4open.science/r/CodeLLM-FD25/.
- **OpenReview**: https://openreview.net/pdf?id=KIPJKST4gw
        
</details>

### Rethinking Channel Dimensions to Isolate Outliers for Low-bit Weight Quantization of Large Language Models
> Per-IC quantization, a novel approach that isolates outliers within input channels, combined with AdaDim, an adaptive framework that accommodates weight sensitivity patterns, significantly improves the efficiency of weight-only quantization in small batch inference settings, with demonstrated enhancements in language modeling benchmarks for both base and instruction-tuned LLMs.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have recently demonstrated a remarkable success across various tasks. However, efficiently serving LLMs has been a challenge due to its large memory bottleneck, specifically in small batch inference settings (e.g. mobile devices). Weight-only quantization can be a promising approach, but sub-4 bit quantization remains a challenge due to large-magnitude activation outliers. To mitigate the undesirable outlier effect, we first propose per-IC quantization, a simple yet effective method that creates quantization groups within each input channel (IC) rather than the conventional per-output channel (OC). Our method is motivated by the observation that activation outliers affect the input dimension of the weight matrix, so similarly grouping the weights in the IC direction can $\\textit{isolate outliers to be within a group}$. We also find that activation outliers do not dictate quantization difficulty, and inherent weight sensitivities also exist. With per-IC quantization as a new outlier-friendly scheme, we then propose Adaptive Dimensions ($\\textbf{AdaDim}$), a versatile quantization framework that can adapt to various weight sensitivity patterns. We demonstrate the effectiveness of AdaDim by augmenting prior methods such as Round-To-Nearest and GPTQ, showing significant improvements across various language modeling benchmarks for both base (up to $+4.7%$ on MMLU) and instruction-tuned (up to $+10%$ on HumanEval) LLMs.
- **OpenReview**: https://openreview.net/pdf?id=JzG7kSpjJk
        
</details>

### Skill-Mix: a Flexible and Expandable Family of Evaluations for AI Models
> The study proposes an evaluation method, Skill-Mix, to assess whether LLMs can flexibly combine learned skills, revealing differences in model capabilities that previous leaderboard evaluations missed.

<details>
<summary>Details</summary>
- **Abstract**: As the role of LLMs shifts from statistical modeling of language to serving as general-purpose AI agents, how should LLM evaluations change? Arguably, a key ability of an AI agent is to flexibly combine, as needed, the basic skills it has learned. This capability to combine skills plays an important role in (human) pedagogy and also in a recent paper on emergence phenomena (Arora & Goyal, 2023). Our paper introduces an evaluation, Skill-Mix, to measure this capability. Using a list of $N$  skills the evaluator repeatedly picks random subsets of  $k$ skills and asks the LLM to produce text combining that subset of skills. Since the number of subsets grows like $N^k$, for even modest $k$ this evaluation will, with high probability, require the LLM to produce text it has not seen in the training set.  The paper develops a methodology for (a) designing and administering such an evaluation, and (b) automatic grading (plus spot-checking by humans) of the results using the open LLaMA-2 70b model as well as  GPT-4.    Administering a version of Skill-Mix to popular chatbots gave results that,  while generally in line with prior expectations, contained surprises. We found sizeable differences in capabilities among models ---including suspected cases of ``cramming for the leaderboard''--- that had not been revealed by the (much simpler) evaluations used in popular LLM leaderboards.  Our methodology can flexibly change to future models and model capabilities, by expanding the set of skills being tested and increasing  $k$. We hope Skill-Mix (which will be publicly released, including all prompts and code) may grow into an eco-system of open evaluations for AI capabilities, including in multi-modal settings.
- **OpenReview**: https://openreview.net/pdf?id=Jf5gplvglq
        
</details>

### Towards Codable Text Watermarking for Large Language Models
> Codable Text Watermarking for Large Language Models (LLMs) enables robust source identification and flexible information encoding, addressing inefficiencies in existing LLM watermarking approaches. The novel Balance-Marking method effectively maintains watermarked text quality while meeting diverse encoding needs.

<details>
<summary>Details</summary>
- **Abstract**: As large language models (LLMs) generate texts with increasing fluency and realism, there is a growing need to identify the source of texts to prevent the abuse of LLMs. Text watermarking techniques have proven reliable in distinguishing whether a text is generated by LLMs by injecting hidden patterns. However, we argue that existing LLM watermarking methods are encoding-inefficient and cannot flexibly meet the diverse information encoding needs (such as encoding model version, generation time, user id, etc.). In this work, we conduct the first systematic study on the topic of **Codable Text Watermarking for LLMs** (CTWL) that allows text watermarks to carry multi-bit customizable information. First of all, we study the taxonomy of LLM watermarking technologies and give a mathematical formulation for CTWL. Additionally, we provide a comprehensive evaluation system for CTWL: (1) watermarking success rate, (2) robustness against various corruptions, (3) coding rate of payload information, (4) encoding and decoding efficiency, (5) impacts on the quality of the generated text. To meet the requirements of these non-Pareto-improving metrics, we follow the most prominent vocabulary partition-based watermarking direction, and devise an advanced CTWL method named **Balance-Marking**. The core idea of our method is to use a proxy language model to split the vocabulary into probability-balanced parts, thereby effectively maintaining the quality of the watermarked text. Extensive experimental results show that our method outperforms the baseline under comprehensive evaluation.
- **OpenReview**: https://openreview.net/pdf?id=JYu5Flqm9D
        
</details>

### Scaling Laws of RoPE-based Extrapolation
> This work highlights the impact of fine-tuning context length and the rotary base value in Rotary Position Embedding-based LLMs on their extrapolation capabilities. It proposes a framework to explain the relationship between these factors and introduces the concept of "critical dimension for extrapolation. Notably, the study demonstrates successful extrapolation up to 1 million context length on LLaMA2 models with limited training length.

<details>
<summary>Details</summary>
- **Abstract**: The extrapolation capability of Large Language Models (LLMs) based on Rotary Position Embedding \\cite{su2021roformer} is currently a topic of considerable interest. The mainstream approach to addressing extrapolation with LLMs involves modifying RoPE by replacing 10000, the rotary base of $\\theta_n={10000}^{-2n/d}$ in the original RoPE, with a larger value and providing longer fine-tuning text. In this work, we first observe that fine-tuning a RoPE-based LLM with either a smaller or larger base in pre-training context length could significantly enhance its extrapolation performance. After that, we propose \\textbf{\\textit{Scaling Laws of RoPE-based Extrapolation}}, a unified framework from the periodic perspective, to describe the relationship between the extrapolation performance and base value as well as tuning context length. In this process, we also explain the origin of the RoPE-based extrapolation issue by \\textbf{\\textit{critical dimension for extrapolation}}. Besides these observations and analyses, we achieve extrapolation up to 1 million context length within only 16K training length on LLaMA2 7B and 13B \\citep{touvron2023llama2}.
- **OpenReview**: https://openreview.net/pdf?id=JO7k0SJ5V6
        
</details>

### Bellman Optimal Step-size Straightening of Flow-Matching Models
> BOSS technique distills flow-matching generative models to achieve efficient image sampling within a resource-constrained budget, optimizing step sizes and refining velocity networks to straighten generation paths, resulting in both resource savings and maintained image quality.

<details>
<summary>Details</summary>
- **Abstract**: Flow matching is a powerful framework for generating high-quality samples in various applications, especially image synthesis. However, the intensive computational demands of these models, especially during the fine-tuning process and sampling processes, pose significant challenges for low-resource scenarios. This paper introduces Bellman Optimal Step-size Straightening (BOSS) technique for distilling flow-matching generative models: it aims specifically for a few-step efficient image sampling while adhering to a computational budget constraint. First, this technique involves a dynamic programming algorithm that optimizes the step sizes of the pretrained network. Then, it refines the velocity network to match the optimal step sizes, aiming to straighten the generation paths. Extensive experimental evaluations across image generation tasks demonstrate the efficacy of BOSS in terms of both resource utilization and image quality. Our results reveal that BOSS achieves substantial gains in efficiency while maintaining competitive sample quality, effectively bridging the gap between low-resource constraints and the demanding requirements of flow-matching generative models. Our paper also fortifies the responsible development of artificial intelligence, offering a more sustainable generative model that reduces computational costs and environmental footprints.
- **OpenReview**: https://openreview.net/pdf?id=Iyve2ycvGZ
        
</details>

### Talk like a Graph: Encoding Graphs for Large Language Models
> The paper examines encoding graph data as text to enhance graph reasoning via large language models (LLMs), revealing impactful factors such as encoding method, graph task nature, and graph structure, which influence LLM performance.

<details>
<summary>Details</summary>
- **Abstract**: Graphs are a powerful tool for representing and analyzing complex relationships in real-world applications such as social networks, recommender systems, and computational finance. Reasoning on graphs is essential for drawing inferences about the relationships between entities in a complex system, and to identify hidden patterns and trends. Despite the remarkable progress in automated reasoning with natural text, reasoning on graphs with large language models (LLMs) remains an understudied problem. In this work, we perform the first comprehensive study of encoding graph-structured data as text for consumption by LLMs. We show that LLM performance on graph reasoning tasks varies on three fundamental levels: (1) the graph encoding method, (2) the nature of the graph task itself, and (3) interestingly, the very structure of the graph considered. These novel results provide valuable insight on strategies for encoding graphs as text. Using these insights we illustrate how the correct choice of encoders can boost performance on graph reasoning tasks inside LLMs by 4.8% to 61.8%, depending on the task.
- **OpenReview**: https://openreview.net/pdf?id=IuXR1CCrSi
        
</details>

### Large Language Models Cannot Self-Correct Reasoning Yet
> LLMs' self-correction abilities are examined in this study, highlighting their potential and limitations. The research finds that LLMs may struggle with self-correction without external feedback, sometimes resulting in performance degradation.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have emerged as a groundbreaking technology with their unparalleled text generation capabilities across various applications. Nevertheless, concerns persist regarding the accuracy and appropriateness of their generated content. A contemporary methodology, self-correction, has been proposed as a remedy to these issues. Building upon this premise, this paper critically examines the role and efficacy of self-correction within LLMs, shedding light on its true potential and limitations. Central to our investigation is the notion of intrinsic self-correction, whereby an LLM attempts to correct its initial responses based solely on its inherent capabilities, without the crutch of external feedback. In the context of reasoning, our research indicates that LLMs struggle to self-correct their responses without external feedback, and at times, their performance might even degrade post self-correction. Drawing from these insights, we offer suggestions for future research and practical applications in this field.
- **OpenReview**: https://openreview.net/pdf?id=IkmD3fKBPQ
        
</details>

### DP-OPT: Make Large Language Model Your Differentially-Private Prompt Engineer
> DP-OPT presents a novel approach for customizing LLMs on sensitive data without compromising privacy by leveraging differentially private prompt generation. This allows for the utilization of cloud-based LLMs while maintaining data privacy, addressing concerns about model ownership and data sharing.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have emerged as dominant tools for various tasks, particularly when tailored for a specific target by prompt tuning. Nevertheless, concerns surrounding data privacy present obstacles when adapting LLMs on sensitive data. A practical solution is to host a local LLM and optimize a soft prompt using private data. Yet, hosting a local model becomes problematic when model ownership is protected. Alternative methods, like sending data to the model's provider for training, intensify these privacy issues. In this paper, we present a novel solution called Differentially-Private Offsite Prompt Tuning (DP-OPT) to address this challenge. Our approach involves tuning a discrete prompt on the client side and then applying it to the desired cloud models. We demonstrate that prompts suggested by LLMs themselves can be transferred without compromising performance much. To ensure that the prompts do not divulge private information, we introduce the first private prompt generation mechanism, by a differentially-private (DP) ensemble of in-context learning with private demonstrations.  With DP-OPT, generating privacy-preserving prompts by Vicuna-7b can yield competitive performance compared to non-private in-context learning on GPT3.5 or local private prompt tuning.
- **OpenReview**: https://openreview.net/pdf?id=Ifz3IgsEPX
        
</details>

### Uncertainty-aware Constraint Inference in Inverse Constrained Reinforcement Learning
> Uncertainty-aware Inverse Constrained Reinforcement Learning (UAICRL) suggests that modeling uncertainties during training is crucial for robust constraint inference and safe control. It incorporates uncertainty quantification into constraint modeling and policy learning, leading to improved performance in environments with stochastic dynamics.

<details>
<summary>Details</summary>
- **Abstract**: Aiming for safe control, Inverse Constrained Reinforcement Learning (ICRL) considers inferring the constraints respected by expert agents from their demonstrations and learning imitation policies that adhere to these constraints. While previous ICRL works often neglected underlying uncertainties during training, we contend that modeling these uncertainties is crucial for facilitating robust constraint inference. This insight leads to the development of an Uncertainty-aware Inverse Constrained Reinforcement Learning (UAICRL) algorithm. Specifically, 1) aleatoric uncertainty arises from the inherent stochasticity of environment dynamics, leading to constraint-violating behaviors in imitation policies. To address this, UAICRL constructs risk-sensitive constraints by incorporating distributional Bellman updates into the cumulative costs model. 2) Epistemic uncertainty, resulting from the model's limited knowledge of Out-of-Distribution (OoD) samples, affects the accuracy of step-wise cost predictions. To tackle this issue, UAICRL develops an information-theoretic quantification of the uncertainty and mitigates its impact through flow-based generative data augmentation. Empirical results demonstrate that UAICRL consistently outperforms other baselines in continuous and discrete environments with stochastic dynamics.
- **OpenReview**: https://openreview.net/pdf?id=ILYjDvUM6U
        
</details>

### Eureka: Human-Level Reward Design via Coding Large Language Models
> Eureka, a reward design algorithm, utilizes the strengths of LLMs to create optimal rewards for complex manipulation tasks. In experiments across 29 RL environments, Eureka surpassed human-crafted rewards, enabling a Shadow Hand simulator to perform pen spinning tricks with remarkable agility.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have excelled as high-level semantic planners for sequential decision-making tasks. However, harnessing them to learn complex low-level manipulation tasks, such as dexterous pen spinning, remains an open problem. We bridge this fundamental gap and present Eureka, a human-level reward design algorithm powered by LLMs. Eureka exploits the remarkable zero-shot generation, code-writing, and in-context improvement capabilities of state-of-the-art LLMs, such as GPT-4, to perform evolutionary optimization over reward code. The resulting rewards can then be used to acquire complex skills via reinforcement learning. Without any task-specific prompting or pre-defined reward templates, Eureka generates reward functions that outperform expert human-engineered rewards. In a diverse suite of 29 open-source RL environments that include 10 distinct robot morphologies, Eureka outperforms human experts on 83% of the tasks, leading to an average normalized improvement of 52%. The generality of Eureka also enables a new gradient-free in-context learning approach to reinforcement learning from human feedback (RLHF), readily incorporating human inputs to improve the quality and the safety of the generated rewards without model updating. Finally, using Eureka rewards in a curriculum learning setting, we demonstrate for the first time, a simulated Shadow Hand capable of performing pen spinning tricks, adeptly manipulating a pen in circles at rapid speed.
- **OpenReview**: https://openreview.net/pdf?id=IEduRUO55F
        
</details>

### Chain-of-Experts: When LLMs Meet Complex Operations Research Problems
> This paper proposes CoE, a new framework that uses multiple agents with different expertise to solve complex OR problems. The agents are orchestrated by a "conductor" that helps them communicate and reason about the problem. Experiments show that CoE outperforms existing LLM-based approaches on both LPWP and ComplexOR, suggesting that it is a promising approach for solving complex OR problems.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have emerged as powerful techniques for various NLP tasks, such as mathematical reasoning and plan generation. In this paper, we study automatic modeling and programming for complex operation research (OR) problems, so as to alleviate the heavy dependence on domain experts and benefit a spectrum of industry sectors. We present the first LLM-based solution, namely Chain-of-Experts (CoE), a novel multi-agent cooperative framework to enhance reasoning capabilities. Specifically, each agent is assigned a specific role and endowed with domain knowledge related to OR. We also introduce a conductor to orchestrate these agents via forward thought construction and backward reflection mechanism. Furthermore, we release a benchmark dataset (ComplexOR) of complex OR problems to facilitate OR research and community development. Experimental results show that CoE significantly outperforms the state-of-the-art LLM-based approaches both on LPWP and ComplexOR.
- **OpenReview**: https://openreview.net/pdf?id=HobyL1B9CZ
        
</details>

### On the Humanity of Conversational AI: Evaluating the Psychological Portrayal of LLMs
> PPBench evaluates psychological aspects of Large Language Models (LLMs), including personality traits, relationships, motivations, and emotions, across various models like ChatGPT and GPT-4, raising questions about the potential emergence of human-like qualities in AI agents.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have recently showcased their remarkable capacities, not only in natural language processing tasks but also across diverse domains such as clinical medicine, legal consultation, and education. LLMs become more than mere applications, evolving into assistants capable of addressing diverse user requests. This narrows the distinction between human beings and artificial intelligence agents, raising intriguing questions regarding the potential manifestation of personalities, temperaments, and emotions within LLMs. In this paper, we propose a framework, PPBench, for evaluating diverse psychological aspects of LLMs. Comprising thirteen scales commonly used in clinical psychology, PPBench further classifies these scales into four distinct categories: personality traits, interpersonal relationships, motivational tests, and emotional abilities. Our study examines five popular models, namely \\texttt{text-davinci-003}, ChatGPT, GPT-4, LLaMA-2-7b, and LLaMA-2-13b. Additionally, we employ a jailbreak approach to bypass the safety alignment protocols and test the intrinsic natures of LLMs. We have made PPBench openly accessible via *\\footnote{The link is hidden due to anonymity. For reviewers, please refer to the supplementary materials.}.
- **OpenReview**: https://openreview.net/pdf?id=H3UayAQWoE
        
</details>

### Tree-Planner: Efficient Close-loop Task Planning with Large Language Models
> Tree-Planner reframes task planning with LLMs into three distinct phases to address the inefficiencies of iterative prompt-based planning, achieving state-of-the-art performance while reducing token consumption by 92.2% and error corrections by 40.5%.

<details>
<summary>Details</summary>
- **Abstract**: This paper studies close-loop task planning, which refers to the process of generating a sequence of skills (a plan) to accomplish a specific goal while adapting the plan based on real-time observations. Recently, prompting Large Language Models (LLMs) to generate actions iteratively has become a prevalent paradigm due to its superior performance and user-friendliness. However, this paradigm is plagued by two inefficiencies: high token consumption and redundant error correction, both of which hinder its scalability for large-scale testing and applications. To address these issues, we propose Tree-Planner, which reframes task planning with LLMs into three distinct phases:  plan sampling,  action tree construction, and grounded deciding. Tree-Planner starts by using an LLM to sample a set of potential plans before execution, followed by the aggregation of them to form an action tree. Finally, the LLM performs a top-down decision-making process on the tree, taking into account real-time environmental information. Experiments show that Tree-Planner achieves state-of-the-art performance while maintaining high efficiency. By decomposing LLM queries into a single plan-sampling call and multiple grounded-deciding calls, a considerable part of the prompt are less likely to be repeatedly consumed.  As a result, token consumption is reduced by 92.2% compared to the previously best-performing model. Additionally, by enabling backtracking on the action tree as needed, the correction process becomes more flexible, leading to a 40.5% decrease in error corrections.
- **OpenReview**: https://openreview.net/pdf?id=Glcsog6zOe
        
</details>

### The Reversal Curse: LLMs trained on A is B fail to learn B is A
> Auto-regressive LLMs struggle with logical deduction, exhibiting a "Reversal Curse" where they fail to generalize statements like "*A is B*" to "*B is A*," despite being prevalent in training data. This failure of logical reasoning is evident in models like GPT-3 and ChatGPT, hindering their ability to answer questions that require reversing the original statement.

<details>
<summary>Details</summary>
- **Abstract**: We expose a surprising failure of generalization in auto-regressive large language models (LLMs). If a model is trained on a sentence of the form "*A is B*", it will not automatically generalize to the reverse direction "*B is A*". This is the **Reversal Curse**. For instance, if a model is trained on "Olaf Scholz was the ninth Chancellor of Germany", it will not automatically be able to answer the question, "Who was the ninth Chancellor of Germany?". Moreover, the likelihood of the correct answer ("Olaf Scholz") will not be higher than for a random name. Thus, models exhibit a basic failure of logical deduction and do not generalize a prevalent pattern in their training set (i.e. if "*A is B*" occurs, "*B is A*" is more likely to occur). We provide evidence for the Reversal Curse by finetuning GPT-3 and Llama-1 on fictitious statements such as "Uriah Hawthorne is the composer of *Abyssal Melodies*" and showing that they fail to correctly answer "Who composed *Abyssal Melodies?*". The Reversal Curse is robust across model sizes and model families and is not alleviated by data augmentation. We also evaluate ChatGPT (GPT-3.5 and GPT-4) on questions about real-world celebrities, such as "Who is Tom Cruise\'s mother? [A: Mary Lee Pfeiffer]" and the reverse "Who is Mary Lee Pfeiffer\'s son?". GPT-4 correctly answers questions like the former 79% of the time, compared to 33% for the latter. This shows a failure of logical deduction that we hypothesize is caused by the Reversal Curse.
- **OpenReview**: https://openreview.net/pdf?id=GPKTIktA0k
        
</details>

### Hypothesis Search: Inductive Reasoning with Language Models
> LLMs' ability to perform inductive reasoning (inferring general principles from specific examples) is enhanced by a pipeline that generates natural language hypotheses at varying levels of abstraction, which are then implemented as Python programs that are verified and generalized to novel inputs. This approach outperforms direct prompting, particularly when incorporating human input in hypothesis selection.

<details>
<summary>Details</summary>
- **Abstract**: Inductive reasoning is a core problem-solving capacity: humans can identify underlying principles from a few examples, which can then be robustly generalized to novel scenarios. Recent work has evaluated large language models (LLMs) on inductive reasoning tasks by directly prompting them yielding "in context learning. This can work well for straightforward inductive tasks, but performs very poorly on more complex tasks such as the Abstraction and Reasoning Corpus (ARC). In this work, we propose to improve the inductive reasoning ability of LLMs by generating explicit hypotheses at multiple levels of abstraction: we prompt the LLM to propose multiple abstract hypotheses about the problem, in natural language, then implement the natural language hypotheses as concrete Python programs. These programs can be directly verified by running on the observed examples and generalized to novel inputs. To reduce the hypothesis search space, we explore steps to filter the set of hypotheses to be implemented as programs: we either ask the LLM to summarize them into a smaller set of hypotheses, or ask human annotators to select a subset. We verify our pipeline\'s effectiveness on the ARC visual inductive reasoning benchmark, its variant 1D-ARC, and string transformation dataset SyGuS. On a random 40-problem subset of ARC, our automated pipeline using LLM summaries achieves 27.5% accuracy, significantly outperforming the direct prompting baseline (accuracy of 12.5%). With the minimal human input of selecting from LLM-generated candidates, the performance is boosted to 37.5%. Our ablation studies show that abstract hypothesis generation and concrete program representations are both beneficial for LLMs to perform inductive reasoning tasks.
- **OpenReview**: https://openreview.net/pdf?id=G7UtIGQmjm
        
</details>

### CRAFT: Customizing LLMs by Creating and Retrieving from Specialized Toolsets
> CRAFT, a flexible framework, equips LLMs with custom task-specific toolsets, enhancing their ability to solve complex tasks by seamlessly integrating external modules, resulting in significant performance improvements.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) are often augmented with tools to solve complex tasks. By generating code snippets and executing them through task-specific Application Programming Interfaces (APIs), they can offload certain functions to dedicated external modules, such as image encoding and performing calculations. However, most existing approaches to augment LLMs with tools are constrained by general-purpose APIs and lack the flexibility for tailoring them to specific tasks. In this work, we present CRAFT, a general tool creation and retrieval framework for LLMs. It creates toolsets specifically curated for the tasks and equips LLMs with a component that retrieves tools from these sets to enhance their capability to solve complex tasks. For each task, we collect specific code solutions by prompting GPT-4 to solve the training examples. Following a validation step ensuring the correctness, these solutions are abstracted into code snippets to enhance reusability, and deduplicated for higher quality. At inference time, the language model retrieves snippets from the toolsets and then executes them or generates the output conditioning on the retrieved snippets. Our method is designed to be flexible and offers a plug-and-play approach to adapt off-the-shelf LLMs to unseen domains and modalities, without any finetuning. Experiments on vision-language, tabular processing, and mathematical reasoning tasks show that our approach achieves substantial improvements compared to strong baselines. In addition, our in-depth analysis reveals that: (1) consistent performance improvement can be achieved by scaling up the number of tools and the capability of the backbone models; (2) each component of our approach contributes to the performance gains; (3) the created tools are well-structured and reliable with low complexity  and atomicity.
- **OpenReview**: https://openreview.net/pdf?id=G0vdDSt9XM
        
</details>

### Does Writing with Language Models Reduce Content Diversity?
> Collaborative writing assisted by large language models raises concerns about diversity loss in the generated content. This study examines the impact of co-writing with GPT3 and InstructGPT on diversity, revealing that InstructGPT leads to reduced diversity due to its less diverse contributions, potentially limiting diverse perspectives in public discourse.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have led to a surge in collaborative writing with model assistance. As different users incorporate suggestions from the same model, there is a risk of decreased diversity in the produced content, potentially limiting diverse perspectives in public discourse. In this work, we measure the impact of co-writing on diversity via a controlled experiment, where users write argumentative essays in three setups---using a base LLM (GPT3), a feedback-tuned LLM (InstructGPT), and writing without model help. We develop a set of diversity metrics and find that writing with InstructGPT (but not the GPT3) results in a statistically significant reduction in diversity. Specifically, it increases the similarity between the writings of different authors and reduces the overall lexical and content diversity. We additionally find that this effect is mainly attributable to InstructGPT contributing less diverse text to co-written essays. In contrast, the user-contributed text remains unaffected by model collaboration. This suggests that the recent improvement in generation quality from adapting models to human feedback might come at the cost of more homogeneous and less diverse content.
- **OpenReview**: https://openreview.net/pdf?id=Feiz5HtCD0
        
</details>

### Alpagasus: Training a Better Alpaca Model with Fewer Data
> Alpagasus, a data selection strategy, identifies low-quality data in instruction-tuning datasets, leading to faster training and improved instruction-following capabilities in large language models. By filtering out misleading data, Alpagasus enables LLMs to achieve performance comparable to their teacher models.

<details>
<summary>Details</summary>
- **Abstract**: Large language models~(LLMs) strengthen instruction-following capability through instruction-finetuning (IFT) on supervised instruction/response data. However, widely used IFT datasets (e.g., Alpaca's 52k data) surprisingly contain many low-quality instances with incorrect or irrelevant responses, which are misleading and detrimental to IFT.  In this paper, we propose a simple and effective data selection strategy that automatically identifies and removes low-quality data using a strong LLM (e.g., ChatGPT). To this end, we introduce Alpagasus, which is finetuned on only 9k high-quality data filtered from the 52k Alpaca data. Alpagasus significantly outperforms the original Alpaca as evaluated by GPT-4 on multiple test sets and the controlled human study. Its 13B variant matches $>90%$ performance of its teacher LLM (i.e., Text-Davinci-003) on test tasks. It also provides 5.7x faster training, reducing the training time for a 7B variant from 80 minutes (for Alpaca) to 14 minutes \\footnote{We apply IFT for the same number of epochs as Alpaca(7B) but on fewer data, using 4$\\times$NVIDIA A100 (80GB) GPUs and following the original Alpaca setting and hyperparameters.}.  In the experiment, we also demonstrate that our method can work not only for machine-generated datasets but also for human-written datasets. Overall, Alpagasus demonstrates a novel data-centric IFT paradigm that can be generally applied to instruction-tuning data, leading to faster training and better instruction-following models.
- **OpenReview**: https://openreview.net/pdf?id=FdVXgSJhvz
        
</details>

### ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
> This paper presents ChatEval, a multi-agent evaluation framework that leverages large language models to provide human-like text assessments. By employing diverse role prompts and encouraging collaborative debate, ChatEval achieves high accuracy and correlation with human evaluations, demonstrating the potential of multi-agent systems for reliable text evaluation.

<details>
<summary>Details</summary>
- **Abstract**: Text evaluation has historically posed significant challenges, often demanding substantial labor and time cost. With the emergence of large language models (LLMs), researchers have explored LLMs' potential as alternatives for human evaluation. While these single-agent-based approaches show promise, experimental results suggest that further advancements are needed to bridge the gap between their current effectiveness and human-level evaluation quality. Recognizing that best practices of human evaluation processes often involve multiple human annotators collaborating in the evaluation, we resort to a multi-agent debate framework, moving beyond single-agent prompting strategies. In this paper, we construct a multi-agent referee team called $\\textbf{ChatEval}$ to autonomously discuss and evaluate the quality of different texts.  Our experiments on two benchmarks illustrate that ChatEval delivers superior accuracy and correlation in alignment with human assessment. Furthermore, we find that the diverse role prompts (different personas) are essential in the multi-agent debate process; that is, utilizing the same role description in the prompts can lead to a degradation in performance. Our qualitative analysis also shows that ChatEval transcends mere textual scoring, offering a human-mimicking evaluation process for reliable assessments.
- **OpenReview**: https://openreview.net/pdf?id=FQepisCUWu
        
</details>

### Bayesian low-rank adaptation for large language models
> Parameter-efficient fine-tuning (PEFT) of large language models (LLMs) with low-rank adaptation (LoRA) suffers from overconfidence, which is addressed in this work by introducing Laplace-LoRA, a Bayesian method that combines the Laplace approximation with LoRA parameters, resulting in improved calibration of fine-tuned LLMs.

<details>
<summary>Details</summary>
- **Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as a new paradigm for cost-efficient fine-tuning of large language models (LLMs), with low-rank adaptation (LoRA) being a widely adopted choice. However, fine-tuned LLMs often become overconfident especially when fine-tuned on small datasets. Bayesian methods, with their inherent ability to estimate uncertainty, serve as potent tools to mitigate overconfidence and enhance calibration. In this work, we introduce Laplace-LoRA, a straightforward yet effective Bayesian method, which applies the Laplace approximation to the LoRA parameters and, considerably boosts the calibration of fine-tuned LLMs.
- **OpenReview**: https://openreview.net/pdf?id=FJiUyzOF1m
        
</details>

### QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models
> QLLM proposes an accurate and efficient low-bitwidth Post-Training Quantization method for Large Language Models (LLMs) by mitigating activation outliers and optimizing tuning methods, demonstrating superior performance on billion-parameter LLaMA models.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have demonstrated unparalleled efficacy in natural language processing. However, their high computational demands and memory overheads hinder their broad deployment. To address this, two quantization strategies emerge, including Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ). For LLMs, the billions of parameters make the QAT impractical due to the prohibitive training cost and thus PTQ becomes more prevalent. In existing studies, activation outliers in particular channels are identified as the biggest challenge to PTQ accuracy. They propose to transform the magnitudes from activations to weights, which however offers limited alleviation or suffers from unstable gradients, resulting in a severe performance drop at low-bitwidth. In this paper, we propose QLLM, an accurate and efficient low-bitwidth PTQ method designed for LLMs. QLLM introduces an adaptive channel reassembly technique that reallocates the magnitude of outliers to other channels, thereby mitigating their impact on the quantization range. This is achieved by channel disassembly and channel assembly, which first breaks down the outlier channels into several sub-channels to ensure a more balanced distribution of activation magnitudes. Then similar channels are merged to maintain the original channel number for efficiency. Additionally, an adaptive strategy is designed to autonomously determine the optimal number of sub-channels for channel disassembly. To further compensate for the performance loss caused by quantization, we propose an efficient tuning method that only learns a small number of low-rank weights while freezing the pre-trained quantized model. After training, these low-rank parameters can be fused into the frozen weights without affecting inference. Extensive experiments on LLaMA-1 and LLaMA-2 show that QLLM is able to obtain accurate quantized models efficiently. For example, QLLM quantizes the 4-bit LLaMA-2-70B within 10 hours on a single A100-80G GPU, outperforming the previous state-of-the-art method by 7.89% on the average accuracy across five zero-shot tasks.
- **OpenReview**: https://openreview.net/pdf?id=FIplmUWdm3
        
</details>

### ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving
> ToRA, a series of tool-integrated agents, combines the power of natural language reasoning with external tools to excel at challenging mathematical problems, significantly outperforming open-source models on diverse reasoning datasets.

<details>
<summary>Details</summary>
- **Abstract**: Large language models have made significant progress in various language tasks, yet they still struggle with complex mathematics. In this paper, we propose ToRA a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools (e.g., computation libraries and symbolic solvers), thereby amalgamating the analytical prowess of language and the computational efficiency of tools. To train ToRA, we curate interactive tool-use trajectories on mathematical datasets, apply imitation learning on the annotations, and propose output space shaping to further refine models' reasoning behavior. As a result, ToRA models significantly outperform open-source models on 10 mathematical reasoning datasets across all scales with 13%-19% absolute improvements on average. Notably, ToRA-7B reaches 44.6% on the competition-level dataset MATH, surpassing the best open-source model WizardMath-70B by 22% absolute. ToRA-34B is also the first open-source model that achieves an accuracy exceeding 50% on MATH, which significantly outperforms GPT-4's CoT result, and is competitive with GPT-4 solving problems with programs. Additionally, we conduct a comprehensive analysis of the benefits and remaining challenges of tool interaction for mathematical reasoning, providing valuable insights for future research.
- **OpenReview**: https://openreview.net/pdf?id=Ep0TtjVoap
        
</details>

### Building Cooperative Embodied Agents Modularly with Large Language Models
> By integrating LLMs' cognitive capabilities into a multi-agent framework, researchers introduce *CoELA*, an agent that can plan, communicate, and cooperate effectively in complex environments, even with limited communication and diverse tasks.

<details>
<summary>Details</summary>
- **Abstract**: In this work, we address challenging multi-agent cooperation problems with decentralized control, raw sensory observations, costly communication, and multi-objective tasks instantiated in various embodied environments.  While previous research either presupposes a cost-free communication channel or relies on a centralized controller with shared observations,  we harness the commonsense knowledge, reasoning ability, language comprehension, and text generation prowess of LLMs and seamlessly incorporate them into a cognitive-inspired modular framework that integrates with perception, memory, and execution. Thus building a **Co**operative **E**mbodied **L**anguage **A**gent *CoELA*, who can plan, communicate, and cooperate with others to accomplish long-horizon tasks efficiently.  Our experiments on C-WAH and TDW-MAT demonstrate that *CoELA* driven by GPT-4 can surpass strong planning-based methods and exhibit emergent effective communication.  Though current Open LMs like LLAMA-2 still underperform, we fine-tune a *CoLLAMA* with data collected with our agents and show how they can achieve promising performance.  We also conducted a user study for human-agent interaction and discovered that *CoELA* communicating in natural language can earn more trust and cooperate more effectively with humans.  Our research underscores the potential of LLMs for future research in multi-agent cooperation. Videos can be found on the project website https://llm-co.github.io/CoELA/ .
- **OpenReview**: https://openreview.net/pdf?id=EnXJfQqy0K
        
</details>

### Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation
> This paper delves into the issue of self-contradiction in language models, demonstrating its prevalence and proposing an innovative detection and mitigation framework that eliminates contradictory content without compromising text quality.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (large LMs) are susceptible to producing text that contains hallucinated content. An important instance of this problem is self-contradiction, where the LM generates two contradictory sentences within the same context. In this work, we present a comprehensive investigation into self-contradiction for various instruction-tuned LMs, covering evaluation, detection, and mitigation. Our analysis reveals the prevalence of self-contradictions when LMs generate text for open-domain topics, e.g., in 17.7% of all sentences produced by ChatGPT. Self-contradiction also complements retrieval-based methods, as a large portion of them (e.g., 35.8% for ChatGPT) cannot be verified using Wikipedia. We then propose a novel prompting-based framework designed to effectively detect and mitigate self-contradictions. Our detector achieves high accuracy, e.g., around 80% F1 score when prompting ChatGPT. The mitigation algorithm iteratively refines the generated text to remove contradictory information while preserving text fluency and informativeness. Importantly, our entire framework is applicable to black-box LMs and does not require external grounded knowledge. Our approach is practically effective and has been released as a push-button tool to benefit the public, with an anonymized version at https://iclr9113.com/.
- **OpenReview**: https://openreview.net/pdf?id=EmQSOi1X2f
        
</details>

### L2MAC: Large Language Model Automatic Computer for Unbounded Code Generation
> This paper introduces L2MAC, a novel memory-augmented LLM that enables long and consistent code generation. L2MAC's unique memory architecture, composed of an instruction registry and file store, allows it to bypass the context window limitation of transformers, enabling the generation of complex code structures that meet user requirements.

<details>
<summary>Details</summary>
- **Abstract**: Transformer-based large language models (LLMs) are constrained by the fixed context window of the underlying transformer architecture, hindering their ability to produce long and logically consistent code. Memory-augmented LLMs are a promising solution, but current approaches cannot handle long code generation tasks since they (1) only focus on reading memory and reduce its evolution to the concatenation of new memories or (2) use very specialized memories that cannot adapt to other domains. This paper presents L2MAC, the first practical LLM-based stored-program automatic computer for long and consistent code generation. Its memory has two components: the instruction registry, which is populated with a prompt program to solve the user-given task, and a file store, which will contain the final and intermediate outputs. Each instruction is executed by a separate LLM instance, whose context is managed by a control unit capable of precise memory reading and writing to ensure effective interaction with the file store. These components enable L2MAC to generate virtually unbounded code structures, bypassing the constraints of the finite context window while producing code that fulfills complex user-specified requirements. We empirically show that L2MAC succeeds in generating large code bases for system design tasks where other coding methods fall short in implementing user requirements and provide insight into the reasons for this performance gap.
- **OpenReview**: https://openreview.net/pdf?id=EhrzQwsV4K
        
</details>

### HyperAttention: Long-context Attention in Near-Linear Time
> The paper introduces HyperAttention, an approximate attention mechanism that overcomes computational challenges of long contexts in Language Models, achieving linear time sampling despite previous complexity limitations.

<details>
<summary>Details</summary>
- **Abstract**: We present an approximate attention mechanism named `HyperAttention` to address the computational challenges posed by the growing complexity of long contexts used in Large Language Models (LLMs).  Recent work suggests that in the worst-case scenario, the quadratic time is necessary unless the entries of the attention matrix are bounded or the matrix has low stable rank.  We introduce two parameters which measure: (1) the max column norm in the normalized attention matrix, and (2) the ratio of row norms in the unnormalized attention matrix after detecting and removing large entries. We use these fine-grained parameters to capture the hardness of the problem.  Despite previous lower bounds, we are able to achieve a linear time sampling algorithm even when the matrix has unbounded entries or a large stable rank, provided the above parameters are small. HyperAttention features a modular design that easily accommodates integration of other fast low-level implementations, particularly FlashAttention.  Empirically, employing Locality Sensitive Hashing (LSH) to identify large entries, HyperAttention outperforms existing methods, giving significant speed improvements compared to state-of-the-art solutions like FlashAttention.  This development presents substantial implications for enabling LLMs to handle significantly larger contexts.
- **OpenReview**: https://openreview.net/pdf?id=Eh0Od2BJIM
        
</details>

### AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors
> The paper proposes a multi-agent framework, AgentVerse, that enables collaborative teamwork among AI agents, leading to improved performance. AgentVerse orchestrates expert agents to act as a cohesive group, demonstrating superior effectiveness in various tasks, from text understanding to reasoning and embodied AI.

<details>
<summary>Details</summary>
- **Abstract**: Autonomous agents empowered by Large Language Models (LLMs) have undergone significant improvements, enabling them to generalize across a broad spectrum of tasks. However, in real-world scenarios, cooperation among individuals is often required to enhance the efficiency and effectiveness of task accomplishment. Hence, inspired by human group dynamics, we propose a multi-agent framework AgentVerse that can effectively orchestrate a collaborative group of expert agents as a greater-than-the-sum-of-its-parts system. Our experiments demonstrate that AgentVerse can proficiently deploy multi-agent groups that outperform a single agent. Extensive experiments on text understanding, reasoning, coding, tool utilization, and embodied AI confirm the effectiveness of AgentVerse. Moreover, our analysis of agent interactions within AgentVerse reveals the emergence of specific collaborative behaviors, contributing to heightened group efficiency. We will release our codebase, AgentVerse, to further facilitate multi-agent research.
- **OpenReview**: https://openreview.net/pdf?id=EHg5GDnyq1
        
</details>

### Vision-by-Language for Training-Free Compositional Image Retrieval
> CIReVL proposes a training-free pipeline for Compositional Image Retrieval (CIR) using pre-trained vision and language models. It achieves competitive performance without requiring costly annotations and enables modular reasoning and scalability.

<details>
<summary>Details</summary>
- **Abstract**: Given an image and a target modification (e.g an image of the Eiffel tower and the text without people and at night-time), Compositional Image Retrieval (CIR) aims to retrieve the relevant target image in a database. While supervised approaches rely on annotating triplets that is costly (i.e. query image, textual modification, and target image), recent research sidesteps this need by using large-scale vision-language models (VLMs), performing Zero-Shot CIR (ZS-CIR). However, state-of-the-art approaches in ZS-CIR still require training task-specific, customized models over large amounts of image-text pairs. In this work, we proposeto tackle CIR in a training-free manner via our Compositional Image Retrieval through Vision-by-Language (CIReVL), a simple, yet human-understandable and scalable pipeline that effectively recombines large-scale VLMs with large language models (LLMs). By captioning the reference image using a pre-trained generative VLM and asking a LLM to recompose the caption based on the textual target modification for subsequent retrieval via e.g. CLIP, we achieve modular language reasoning. In four ZS-CIR benchmarks, we find competitive, in-part state-of-the-art performance - improving over supervised methods Moreover, the modularity of CIReVL offers simple scalability without re-training, allowing us to both investigate scaling laws and bottlenecks for ZS-CIR while easily scaling up to in parts more than double of previously reported results. Finally, we show that CIReVL makes CIR human-understandable by composing image and text in a modular fashion in the language domain, thereby making it intervenable, allowing to post-hoc re-align failure cases. Code will be released upon acceptance.
- **OpenReview**: https://openreview.net/pdf?id=EDPxCjXzSb
        
</details>

### Group Preference Optimization: Few-Shot Alignment of Large Language Models
> GPO, a novel framework, steers language models to align with the preferences of various groups by leveraging meta-learning to train an additional transformer module that predicts group preferences, leading to accurate alignment with fewer preferences and reduced computational requirements.

<details>
<summary>Details</summary>
- **Abstract**: Many applications of large language models (LLMs), ranging from chatbots to creative writing, require nuanced subjective judgments that can differ significantly across different groups. Existing alignment algorithms can be expensive to align for each group, requiring prohibitive amounts of group-specific preference data and computation for real-world use cases. We introduce Group Preference Optimization (GPO), an alignment framework that steers language models to preferences of individual groups in a few-shot manner. In GPO, we augment the base LLM with an independent transformer module trained to predict the preferences of a group for the LLM generations. For few-shot learning, we parameterize this module as an in-context autoregressive transformer and train it via meta-learning on several groups. We empirically validate the efficacy of GPO through rigorous evaluations using LLMs with varied sizes on three human opinion adaptation tasks. These tasks involve adapting to the preferences of US demographic groups, global countries, and individual users. Our results demonstrate that GPO not only aligns models more accurately but also requires fewer group-specific preferences and less training and inference computing resources, outperforming existing strategies such as in-context steering and fine-tuning methods.
- **OpenReview**: https://openreview.net/pdf?id=DpFeMH4l8Q
        
</details>

### Constrained Decoding for Cross-lingual Label Projection
> This work presents a novel label projection method for zero-shot cross-lingual transfer learning that overcomes translation quality degradation issues faced by existing marker-based approaches. By leveraging constrained decoding, the method preserves the quality of translated texts and can be applied to both training and test data, boosting performance on NER and Event Argument Extraction tasks across 20 languages.

<details>
<summary>Details</summary>
- **Abstract**: Zero-shot cross-lingual transfer utilizing multilingual LLMs has become a popular learning paradigm for low-resource languages with no labeled training data. However, for NLP tasks that involve fine-grained predictions on words and phrases, the performance of zero-shot cross-lingual transfer learning lags far behind supervised fine-tuning methods. Therefore, it is common to exploit translation and label projection to further improve the performance by (1) translating training data that is available in a high-resource language (e.g., English) together with the gold labels into low-resource languages, and/or (2) translating test data in low-resource languages to a high-source language to run inference on, then projecting the predicted span-level labels back onto the original test data. However, state-of-the-art marker-based label projection methods suffer from translation quality degradation due to the extra label markers injected in the input to the translation model. In this work, we explore a new direction that leverages constrained decoding for label projection to overcome the aforementioned issues. Our new method not only can preserve the quality of translated texts but also has the versatility of being applicable to both translating training and translating test data strategies. This versatility is crucial as our experiments reveal that translating test data can lead to a considerable boost in performance compared to translating only training data. We evaluate on two cross-lingual transfer tasks, namely Named Entity Recognition and Event Argument Extraction, spanning 20 languages. The results demonstrate that our approach outperforms the state-of-the-art marker-based method by a large margin and also shows better performance than other label projection methods that rely on external word alignment.
- **OpenReview**: https://openreview.net/pdf?id=DayPQKXaQk
        
</details>

### On the Reliability of Watermarks for Large Language Models
> Watermarking effectively detects LLM-generated text, even after human or LLM-based rewriting, as it remains detectable within a range of text lengths and even when embedded in longer documents.

<details>
<summary>Details</summary>
- **Abstract**: As LLMs become commonplace, machine-generated text has the potential to flood the internet with spam, social media bots, and valueless content. _Watermarking_ is a simple and effective strategy for mitigating such harms by enabling the detection and documentation of LLM-generated text. Yet a crucial question remains: How reliable is watermarking in realistic settings in the wild? There, watermarked text may be modified to suit a user's needs, or entirely rewritten to avoid detection. We study the robustness of watermarked text after it is re-written by humans, paraphrased by a non-watermarked LLM, or mixed into a longer hand-written document. We find that watermarks remain detectable even after human and machine paraphrasing. While these attacks dilute the strength of the watermark, paraphrases are statistically likely to leak n-grams or even longer fragments of the original text, resulting in high-confidence detections when enough tokens are observed.  For example, after strong human paraphrasing the watermark is detectable after observing 800 tokens on average, when setting a $1\\mathrm{e}{-5}$ false positive rate. We also consider a range of new detection schemes that are sensitive to short spans of watermarked text embedded inside a large document, and we compare the robustness of watermarking to other kinds of detectors.
- **OpenReview**: https://openreview.net/pdf?id=DEJIDCmWOz
        
</details>

### WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions
> This study introduces Evol-Instruct, an automated method for generating instruction data of varying complexity. By using an LLM to iteratively rewrite initial instructions, Evol-Instruct creates a large dataset that outperforms baselines in training LLMs like WizardLM, suggesting the potential of automated instruction creation.

<details>
<summary>Details</summary>
- **Abstract**: Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Both automatic and human evaluations consistently indicate that WizardLM outperforms baselines such as Alpaca (trained from Self-Instruct) and Vicuna (trained from human-created instructions). The experimental results demonstrate that the quality of instruction-following dataset crafted by Evol-Instruct can significantly improve the performance of LLMs.
- **OpenReview**: https://openreview.net/pdf?id=CfXh93NDgH
        
</details>

### FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets
> The FLASK (Fine-grained Language Model Evaluation based on Alignment Skill Sets) protocol allows for more precise assessment of LLMs by breaking down overall scoring into skill-specific assessments for each instruction. This fine-grained approach helps uncover model strengths and weaknesses and improves the reliability of evaluations.

<details>
<summary>Details</summary>
- **Abstract**: Evaluation of Large Language Models (LLMs) is challenging because instruction-following necessitates alignment with human values and the required set of skills varies depending on the instruction. However, previous studies have mainly focused on coarse-grained evaluation (i.e. overall preference-based evaluation), which limits interpretability since it does not consider the nature of user instructions that require instance-wise skill composition. In this paper, we introduce FLASK (Fine-grained Language Model Evaluation based on Alignment Skill Sets), a fine-grained evaluation protocol for both human-based and model-based evaluation which decomposes coarse-level scoring to a skill set-level scoring for each instruction. We experimentally observe that the fine-graininess of evaluation is crucial for attaining a holistic view of model performance and increasing the reliability of the evaluation. Using FLASK, we compare multiple open-source and proprietary LLMs and observe a high correlation between model-based and human-based evaluations.
- **OpenReview**: https://openreview.net/pdf?id=CYmF38ysDa
        
</details>

### LLM-CXR: Instruction-Finetuned LLM for CXR Image Understanding and Generation
> The paper introduces a method for instructing a large language model to enhance its vision-language capabilities specifically for medical images, enabling it to understand and generate both text and images seamlessly, thus improving the model's alignment and reasoning abilities in both understanding and generating medical images like Chest X-rays.

<details>
<summary>Details</summary>
- **Abstract**: Following the impressive development of LLMs, vision-language alignment in LLMs is actively being researched to enable multimodal reasoning and visual input/output. This direction of research is particularly relevant to medical imaging because accurate medical image analysis and generation consist of a combination of reasoning based on visual features and prior knowledge. Many recent works have focused on training adapter networks that serve as an information bridge between image processing (encoding or generating) networks and LLMs; but presumably, in order to achieve maximum reasoning potential of LLMs on visual information as well, visual and language features should be allowed to interact more freely. This is especially important in the medical domain because understanding and generating medical images such as chest X-rays (CXR) require not only accurate visual and language-based reasoning but also a more intimate mapping between the two modalities. Thus, taking inspiration from previous work on the transformer and VQ-GAN combination for bidirectional image and text generation, we build upon this approach and develop a method for instruction-tuning an LLM pre-trained only on text to gain vision-language capabilities for medical images. Specifically, we leverage a pretrained LLMs existing question-answering and instruction-following abilities to teach it to understand visual inputs by instructing it to answer questions about image inputs and, symmetrically, output both text and image responses appropriate to a given query by tuning the LLM with diverse tasks that encompass image-based text-generation and text-based image-generation. We show that our LLM-CXR trained in this approach shows better image-text alignment in both CXR understanding and generation tasks while being smaller in size compared to previously developed models that perform a narrower range of tasks.
- **OpenReview**: https://openreview.net/pdf?id=BqHaLnans2
        
</details>

### Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature
> Fast-DetectGPT uses a novel metric to efficiently detect if content is machine- or human-generated, surpassing the performance of existing detectors while significantly reducing computational costs.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have shown the ability to produce fluent and cogent content, presenting both productivity opportunities and societal risks. To build trustworthy AI systems, it is imperative to distinguish between machine-generated and human-authored content. The leading zero-shot detector, DetectGPT, showcases commendable performance but is marred by its intensive computational  costs. In this paper, we introduce the concept of \\emph{conditional probability curvature} to elucidate discrepancies in word choices between LLMs and humans within a given context. Utilizing this curvature as a foundational metric, we present \\emph{Fast-DetectGPT}, an optimized zero-shot detector, which substitutes DetectGPT's perturbation step with a more efficient sampling step. Our evaluations on various datasets, source models, and test conditions indicate that Fast-DetectGPT not only outperforms DetectGPT in both the white-box and black-box settings but also accelerates the detection process by a factor of 340, as detailed in Table 1.
- **OpenReview**: https://openreview.net/pdf?id=Bpcgcr8E8Z
        
</details>

### PB-LLM: Partially Binarized Large Language Models
> PB-LLM, a novel approach, partially-binarizes significant weights in large language models (LLMs) for extreme low-bit quantization, maintaining their linguistic reasoning abilities through Hessian-guided weight reconstruction and optimal scaling during training.

<details>
<summary>Details</summary>
- **Abstract**: This paper explores network binarization, a radical form of quantization, compressing model weights to a single bit, specifically for Large Language Models (LLMs) compression.  Due to previous binarization methods collapsing LLMs, we propose a novel approach, Partially-Binarized LLM (PB-LLM), which can achieve extreme low-bit quantization while maintaining the linguistic reasoning capacity of quantized LLMs.  Specifically, our exploration first uncovers the ineffectiveness of naive applications of existing binarization algorithms and highlights the imperative role of salient weights in achieving low-bit quantization.  Thus, PB-LLM filters a small ratio of salient weights during binarization, allocating them to higher-bit storage, i.e., partially-binarization.  PB-LLM is extended to recover the capacities of quantized LMMs, by analyzing from the perspective of post-training quantization (PTQ) and quantization-aware training (QAT).  Under PTQ, combining the concepts from GPTQ, we reconstruct the binarized weight matrix guided by the Hessian matrix and successfully recover the reasoning capacity of PB-LLM in low-bit.  Under QAT, we freeze the salient weights during training, explore the derivation of optimal scaling factors crucial for minimizing the quantization error, and propose a scaling mechanism based on this derived scaling strategy for residual binarized weights.  Those explorations and the developed methodologies significantly contribute to rejuvenating the performance of low-bit quantized LLMs and present substantial advancements in the field of network binarization for LLMs.
- **OpenReview**: https://openreview.net/pdf?id=BifeBRhikU
        
</details>

### Large Language Models as Optimizers
> OPRO proposes a novel approach to optimization by utilizing large language models (LLMs) to generate solutions and leverage their language generation capabilities. Through a series of optimization steps, the LLM generates new solutions based on previously generated ones, resulting in the discovery of improved prompt instructions for various optimization tasks.

<details>
<summary>Details</summary>
- **Abstract**: Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to prompt optimization where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks.
- **OpenReview**: https://openreview.net/pdf?id=Bb4VGOWELI
        
</details>

### AntGPT: Can Large Language Models Help Long-term Action Anticipation from Videos?
> Large language models (LLMs) show promise for long-term action anticipation (LTA) by providing priors for possible actions and inferring goals. AntGPT, which uses LLMs to infer goals and model temporal dynamics, excels in LTA tasks, suggesting that LLMs' capabilities can be distilled into compact models.

<details>
<summary>Details</summary>
- **Abstract**: Can we better anticipate an actors future actions (e.g. mix eggs) by knowing what commonly happens after the current action (e.g. crack eggs)? What if the actor also shares the goal (e.g. make fried rice) with us? The long-term action anticipation (LTA) task aims to predict an actors future behavior from video observations in the form of verb and noun sequences, and it is crucial for human-machine interaction. We propose to formulate the LTA task from two perspectives: a bottom-up approach that predicts the next actions autoregressively by modeling temporal dynamics; and a top-down approach that infers the goal of the actor and plans the needed procedure to accomplish the goal. We hypothesize that large language models (LLMs), which have been pretrained on procedure text data (e.g. recipes, how-tos), have the potential to help LTA from both perspectives. It can help provide the prior knowledge on the possible next actions, and infer the goal given the observed part of a procedure, respectively. We propose AntGPT, which represents video observations as sequences of human actions, and uses the action representation for an LLM to infer the goals and model temporal dynamics. AntGPT achieves state- of-the-art performance on Ego4D LTA v1 and v2, EPIC-Kitchens-55, as well as EGTEA GAZE+, thanks to LLMs goal inference and temporal dynamics modeling capabilities. We further demonstrate that these capabilities can be effectively distilled into a compact neural network 1.3% of the original LLM model size. Code and model will be released upon acceptance.
- **OpenReview**: https://openreview.net/pdf?id=Bb21JPnhhr
        
</details>

### Fine-tuning Multimodal LLMs to Follow Zero-shot Demonstrative Instructions
> This study introduces VPG-C, which complements MLLMs in comprehending complex demonstrative instructions by inferring missing visual details. Trained synthetically, VPG-C outperforms existing methods in understanding these instructions.

<details>
<summary>Details</summary>
- **Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs) have been utilizing Visual Prompt Generators (VPGs) to convert visual features into tokens that LLMs can recognize. This is achieved by training the VPGs on millions of image-caption pairs, where the VPG-generated tokens of images are fed into a frozen LLM to generate the corresponding captions. However, this image-captioning based training objective inherently biases the VPG to concentrate solely on the primary visual contents sufficient for caption generation, often neglecting other visual details. This shortcoming results in MLLMs underperformance in comprehending demonstrative instructions consisting of multiple, interleaved, and multimodal instructions that demonstrate the required context to complete a task. To address this issue, we introduce a generic and lightweight Visual Prompt Generator Complete module (VPG-C), which can infer and complete the missing details essential for comprehending demonstrative instructions. Further, we propose a synthetic discriminative training strategy to fine-tune VPG-C, eliminating the need for supervised demonstrative instructions. As for evaluation, we build DEMON, a comprehensive benchmark for demonstrative instruction understanding. Synthetically trained with the proposed strategy, VPG-C achieves significantly stronger zero-shot performance across all tasks of DEMON. Further evaluation on the MME and OwlEval benchmarks also demonstrate the superiority of VPG-C. The anonymous project is available at https://anonymous.4open.science/r/Cheetah-45B4.
- **OpenReview**: https://openreview.net/pdf?id=BXY6fe7q31
        
</details>

### RealChat-1M: A Large-Scale Real-World LLM Conversation Dataset
> RealChat-1M, a large dataset of real-world conversations with LLMs, offers insights into human-LM interactions. This dataset enables the development of content moderation models, safety benchmarks, instruction-following models, and challenging benchmark questions, highlighting the dataset's versatility and potential to advance LLM capabilities.

<details>
<summary>Details</summary>
- **Abstract**: Studying how people interact with large language models (LLMs) in real-world scenarios is increasingly important due to their widespread use in various applications. In this paper, we introduce RealChat-1M, a large-scale dataset containing one million real-world conversations with 25 state-of-the-art LLMs. This dataset is collected from 210K unique IP addresses in the wild on our chat demo website. We offer an overview of the dataset's content, including its curation process, basic statistics, and topic distribution, highlighting its diversity, originality, and scale. We demonstrate its versatility through four use cases: developing content moderation models that perform similarly to GPT-4, building a safety benchmark, training instruction-following models that perform similarly to Vicuna, and creating challenging benchmark questions. We believe that this dataset will serve as a valuable resource for understanding and advancing LLM capabilities. The dataset will be publicly available.
- **OpenReview**: https://openreview.net/pdf?id=BOfDKxfwt0
        
</details>

### Compressing LLMs: The Truth is Rarely Pure and Never Simple
> Existing compression approaches for LLMs, while promising, may fail to accurately preserve their capabilities. LLM-KICK, a rigorous evaluation benchmark, reveals the limitations of pruning methods and the relative effectiveness of quantization while highlighting the robustness of pruned LLMs in certain applications.

<details>
<summary>Details</summary>
- **Abstract**: Despite their remarkable achievements, modern Large Language Models (LLMs) encounter exorbitant computational and memory footprints. Recently, several works have shown significant success in *training-free* and  *data-free* compression (pruning and quantization) of LLMs achieving 50-60% sparsity and reducing the bit-width down to 3 or 4 bits per weight, with negligible perplexity degradation over the uncompressed baseline. As recent research efforts are focused on developing increasingly sophisticated compression methods, our work takes a step back, and re-evaluates the effectiveness of existing SoTA compression methods, which rely on a fairly simple and widely questioned metric, perplexity (even for dense LLMs). We introduce **K**nowledge-**I**ntensive **C**ompressed LLM Benchmar**K** **(LLM-KICK)**, a collection of carefully-curated tasks to re-define the evaluation protocol for compressed LLMs, which have significant alignment with their dense counterparts, and perplexity fail to capture subtle change in their true capabilities. LLM-KICK unveils many favorable merits and unfortunate plights of current SoTA compression methods: all pruning methods suffer significant performance degradation, sometimes at trivial sparsity ratios (*e.g.*, 25-30%), and fail for N:M sparsity on knowledge-intensive tasks; current quantization methods are more successful than pruning; yet, pruned LLMs even at $\\geq 50$% sparsity are robust in-context retrieval and summarization systems; among others. LLM-KICK is designed to holistically access compressed LLMs' ability for language understanding, reasoning, generation, in-context retrieval, in-context summarization, *etc.* We hope our study can foster the development of better LLM compression methods. All our related codes are planed to be open-sourced.
- **OpenReview**: https://openreview.net/pdf?id=B9klVS7Ddk
        
</details>

### ToolChain*: Efficient Action Space Navigation in Large Language Models with A* Search
> ToolChain* is an efficient planning algorithm for AI agents that use large language models (LLMs). It enhances decision-making by navigating expansive action spaces through an efficient tree search, balancing exploration and exploitation to find optimal solutions with less time.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have demonstrated powerful decision-making and planning capabilities in solving complicated real-world problems. LLM-based autonomous agents can interact with diverse tools (e.g., functional APIs) and generate solution plans that execute a series of API function calls in a step-by-step manner. The multitude of candidate API function calls significantly expands the action space, amplifying the critical need for efficient action space navigation. However, existing methods either struggle with unidirectional exploration in expansive action spaces, trapped into a locally optimal solution, or suffer from exhaustively traversing all potential actions, causing inefficient navigation. To address these issues, we propose ToolChain*, an efficient tree search-based planning algorithm for LLM-based agents. It formulates the entire action space as a decision tree, where each node represents a possible API function call involved in a solution plan. By incorporating the A$^*$ search algorithm with task-specific cost function design, it efficiently prunes high-cost branches that may involve incorrect actions, identifying the most low-cost valid path as the solution. Extensive experiments on multiple tool-use and reasoning tasks demonstrate that ToolChain* efficiently balances exploration and exploitation within an expansive action space. It outperforms state-of-the-art baselines on planning and reasoning tasks by 3.1% and 3.5% on average while requiring 7.35x and 2.31x less time, respectively.
- **OpenReview**: https://openreview.net/pdf?id=B6pQxqUcT8
        
</details>

### LLMs Represent Contextual Tasks as Compact Function Vectors
> Autoregressive language models incorporate simple mechanisms that represent input-output functions as vectors, enabling them to perform a wide range of tasks based on causal effects triggered by these function vectors.

<details>
<summary>Details</summary>
- **Abstract**: We report the presence of a simple mechanism that represents an input-output function as a vector within autoregressive transformer language models. Using causal mediation analysis on a diverse range of in-context-learning (ICL) tasks, we find that a small number attention heads transport a compact representation of the demonstrated task, which we call a function vector (FV). We test the causal effects of FVs in a variety of input contexts and find that for many tasks FVs are robust to changes in context, i.e., they trigger execution of the task on inputs such as zero-shot and natural text settings that do not resemble ICL. By measuring the causal effects of the FV at each layer of the network, we find that FVs do not directly perform a task through embedding arithmetic, but rather they trigger the model to perform the task using potentially nonlinear computations. Finally, we investigate the internal structure of FVs and find while that they contain information that directly encodes the output space of the function, this information alone is not sufficient to reconstruct an FV. Taken together, our findings suggest that LLMs contain internal abstractions of general-purpose functions that can be invoked in a variety of contexts.
- **OpenReview**: https://openreview.net/pdf?id=AwyxtyMwaG
        
</details>

### Understanding Length Generalization by Thinking Like Transformers
> Transformers' length generalization abilities vary across tasks, with strong performance on algorithmic tasks where simple RASP-L programs exist. This suggests a possible relationship between RASP-simplicity and generalization.

<details>
<summary>Details</summary>
- **Abstract**: Large language models exhibit surprising emergent generalization properties, yet also struggle on many simple reasoning tasks such as arithmetic and parity. In this work, we focus on length generalization, and we propose a unifying framework to understand when and how Transformers can be expected to length generalize on a given task. First, we show that there exist algorithmic tasks for which standard decoder-only Transformers trained from scratch naturally exhibit strong length generalization. For these tasks, we leverage the RASP programming language (Weiss et al., 2021) to show that the correct algorithmic solution which solves the task can be represented by a simple Transformer. We thus propose the RASP-Generalization Conjecture: Transformers tend to learn a length-generalizing solution if there exists a short RASP-L program that works for all input lengths. We present empirical evidence to support the correlation between RASP-simplicity and generalization. We leverage our insights to give new scratchpad formats which yield strong length generalization on traditionally hard tasks (such as parity and addition), and we illustrate how scratchpad can hinder generalization when it increases the complexity of the corresponding RASP-L program. Overall, our work provides a novel perspective on the mechanisms of length generalization and the algorithmic capabilities of Transformers.
- **OpenReview**: https://openreview.net/pdf?id=AssIuHnmHX
        
</details>

### KoLA: Carefully Benchmarking World Knowledge of Large Language Models
> The Knowledge-oriented LLM Assessment (KoLA) benchmark proposes a meticulous evaluation approach that incorporates a comprehensive ability taxonomy, diverse data sources (including unseen and evolving knowledge), and contrastive metrics for assessing knowledge-creating abilities of large language models (LLMs).

<details>
<summary>Details</summary>
- **Abstract**: The unprecedented performance of large language models (LLMs) necessitates improvements in evaluations. Rather than merely exploring the breadth of LLM abilities, we believe meticulous and thoughtful designs are essential to thorough, unbiased, and applicable evaluations. Given the importance of world knowledge to LLMs, we construct a Knowledge-oriented LLM Assessment benchmark (KoLA), in which we carefully design three crucial factors: (1) For ability modeling, we mimic human cognition to form a four-level taxonomy of knowledge-related abilities, covering 19 tasks. (2) For data, to ensure fair comparisons, we use both Wikipedia, a corpus prevalently pre-trained by LLMs, along with continuously collected emerging corpora, aiming to evaluate the capacity to handle unseen data and evolving knowledge. (3) For evaluation criteria, we adopt a contrastive system, including overall standard scores for better numerical comparability across tasks and models, and a unique self-contrast metric for automatically evaluating knowledge-creating ability. We evaluate 21 open-source and commercial LLMs and obtain some intriguing findings. The KoLA dataset will be updated every three months to provide timely references for developing LLMs and knowledge-related systems.
- **OpenReview**: https://openreview.net/pdf?id=AqN23oqraW
        
</details>

### BatchPrompt: Accomplish more with less
> This paper introduces BatchPrompt, a strategy of batching multiple data points into a single prompt for LLMs, to improve token utilization compared to regular prompting methods. To address performance degradation in batch prompting, Batch Permutation and Ensembling (BPE) is proposed to recover labeling quality by repeatedly permuting data positions and using majority voting, while Self-reflection-guided Early Stopping (SEAS) is designed to terminate the voting process early for data points handled confidently by the LLM.

<details>
<summary>Details</summary>
- **Abstract**: The ever-increasing token limits of large language models (LLMs) have enabled long context as input. Many LLMs are trained/fine-tuned to perform zero-shot/few-shot inference using instruction-based prompts. Crafting prompts for these LLMs typically requires the user to provide a detailed task description, demonstrations, and single example of context for inference. This regular prompt baseline is referred to as SinglePrompt in this paper. However, for NLP tasks where each data point for inference is not necessarily lengthy, the token count for instructions and few-shot examples in the prompt may be considerably larger than that of the data point, resulting in lower token-resource utilization compared with encoder-based models like fine-tuned BERT. This cost-efficiency issue, affecting inference speed and compute budget, counteracts the many benefits LLMs have to offer. This paper aims to alleviate the preceding problem by batching multiple data points into a single prompt, a prompting strategy we refer to as BatchPrompt. This strategy increases the density of data points, which in turn leads to improved token utilization. Applying BatchPrompt na vely, however, is very challenging due to significant performance degradation, as observed in our experiments. We also noticed varying inference outcomes for the same data points appearing in different positions within a prompt. Based on this observation, to address the quality issue while remain high token-resource utilization, we introduce Batch Permutation and Ensembling (BPE) for BatchPrompt, a simple majority voting way that recovers labeling quality through repeatedly permutating data positions in a batch at the price of more token usage. To counterbalance the additional token usage caused by the voting process, we further propose Self-reflection-guided EArly Stopping (SEAS), which can terminate the voting process early for data points the LLM confidently handles. Our comprehensive experimental evaluation demonstrates that BPE +SEAS can boost the performance of BatchPrompt with a striking margin on a range of popular NLP tasks, including question answering (Boolq), textual entailment (RTE), and duplicate questions identification (QQP). These performances are even competitive with/higher than single-data prompting (SinglePrompt), while BatchPrompt requires much fewer LLM calls and input tokens (For SinglePrompt v.s. BatchPrompt+BPE +SEAS with batch size 32, using just 15.7% the number of LLM calls, Boolq accuracy 90.6%  90.9% with 27.4% tokens, QQP accuracy 87.2%  88.4% with 18.6% tokens, RTE accuracy 91.5%  91.1% with 30.8% tokens). We hope our simple yet effective approach will shed light on the future research of large language models. The code will be released.
- **OpenReview**: https://openreview.net/pdf?id=Agyicd577r
        
</details>

### Unleashing the Power of Pre-trained Language Models for Offline Reinforcement Learning
> LaMo framework leverages pre-trained Large Language Models (LLMs) for offline reinforcement learning, combining the strong language understanding of LLMs with the decision-making capabilities of Decision Transformers. This framework includes innovative techniques such as LoRA fine-tuning and non-linear MLP transformations, resulting in impressive performance in sparse-reward tasks and a competitive edge in dense-reward tasks, particularly in data-limited scenarios.

<details>
<summary>Details</summary>
- **Abstract**: Offline reinforcement learning (RL) aims to find a near-optimal policy using pre-collected datasets. Given recent advances in Large Language Models (LLMs) and their few-shot learning prowess, this paper introduces $\\textbf{La}$nguage Models for $\\textbf{Mo}$tion Control ($\\textbf{LaMo}$), a general framework based on Decision Transformers to effectively use pre-trained Language Models (LMs) for offline RL. Our framework highlights four crucial components: (1)  Initializing Decision Transformers with sequentially pre-trained LMs, (2) employing the LoRA fine-tuning method, in contrast to full-weight fine-tuning, to combine the pre-trained knowledge from LMs and in-domain knowledge effectively, (3) using the non-linear MLP transformation instead of linear projections, to generate embeddings, and (4) integrating an auxiliary language prediction loss during fine-tuning to stabilize the LMs and retain their original abilities on languages. Empirical results indicate $\\textbf{LaMo}$ achieves state-of-the-art performance in sparse-reward tasks and closes the gap between value-based offline RL methods and decision transformers in dense-reward tasks. In particular, our method demonstrates superior performance in scenarios with limited data samples.
- **OpenReview**: https://openreview.net/pdf?id=AY6aM13gGF
        
</details>

### LoTa-Bench: Benchmarking Language-oriented Task Planners for Embodied Agents
> This paper presents a benchmark system for evaluating language-oriented task planners in home-service scenarios, allowing for easy comparison of different models and prompts. Through extensive experimentation, the study examines the impact of model selection and prompt construction, providing insights for improving planner performance.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have recently received considerable attention as alternative solutions for task planning. However, comparing the performance of language-oriented task planners becomes difficult, and there exists a dearth of detailed exploration regarding the effects of various factors such as pre-trained model selection and prompt construction. To address this, we propose a benchmark system for automatically quantifying performance of task planning for home-service embodied agents. Task planners are tested on two pairs of datasets and simulators: 1) ALFRED and AI2-THOR, 2) an extension of Watch-And-Help and VirtualHome. Using the proposed benchmark system, we perform extensive experiments with LLMs and prompts, and explore several enhancements of the baseline planner. We expect that the proposed benchmark tool would accelerate the development of language-oriented task planners.
- **OpenReview**: https://openreview.net/pdf?id=ADSxCpCu9s
        
</details>

### On Double-Descent in Reinforcement Learning with LSTD and Random Features
> This paper investigates the effects of neural network size and regularization in Deep RL, identifying the over-parameterization regime as crucial and observing a double-descent phenomenon in performance, attributed to correction terms vanishing with increased regularization or fewer unvisited states.

<details>
<summary>Details</summary>
- **Abstract**: Temporal Difference (TD) algorithms are widely used in Deep Reinforcement Learning (RL). Their performance is heavily influenced by the size of the neural network. While in supervised learning, the regime of over-parameterization and its benefits are well understood, the situation in RL is much less clear. In this paper, we present a theoretical analysis of the influence of network size and $l_2$-regularization on performance. We identify the ratio between the number of parameters and the number of visited states as a crucial factor and define over-parameterization as the regime when it is larger than one. Furthermore, we observe a double-descent phenomenon, i.e., a sudden drop in performance around the parameter/state ratio of one. Leveraging random features and the lazy training regime, we study the regularized Least-Square Temporal Difference (LSTD) algorithm in an asymptotic regime, as both the number of parameters and states go to infinity, maintaining a constant ratio. We derive deterministic limits of both the empirical and the true Mean-Square Bellman Error (MSBE) that feature correction terms responsible for the double-descent. Correction terms vanish when the $l_2$-regularization is increased or the number of unvisited states goes to zero. Numerical experiments with synthetic and small real-world environments closely match the theoretical predictions.
- **OpenReview**: https://openreview.net/pdf?id=9RIbNmx984
        
</details>

### Do Large Language Models Know about Facts?
> Pinocchio, a comprehensive benchmark, evaluates the factual knowledge of LLMs, revealing that they still lack in this area despite their impressive language processing abilities, hindering the development of trustworthy AI.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have recently driven striking performance improvements across a range of natural language processing tasks. The factual knowledge acquired during pretraining and instruction tuning can be useful in various downstream tasks, such as question answering, and language generation. Unlike conventional Knowledge Bases (KBs) that explicitly store factual knowledge, LLMs implicitly store facts in their parameters. Content generated by the LLMs can often exhibit inaccuracies or deviations from the truth, due to facts that can be incorrectly induced or become obsolete over time. To this end, we aim to comprehensively evaluate the extent and scope of factual knowledge within LLMs by designing the benchmark Pinocchio. Pinocchio contains 20K diverse factual questions that span different sources, timelines, domains, regions, and languages. Furthermore, we investigate whether LLMs are able to compose multiple facts, update factual knowledge temporally, reason over multiple pieces of facts, identify subtle factual differences, and resist adversarial examples. Extensive experiments on different sizes and types of LLMs show that existing LLMs still lack factual knowledge and suffer from various spurious correlations. We believe this is a critical bottleneck for realizing trustworthy artificial intelligence. The dataset Pinocchio and our codes will be publicly available.
- **OpenReview**: https://openreview.net/pdf?id=9OevMUdods
        
</details>

### A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis
> WebAgent, an LLM agent powered by Flan-U-PaLM and HTML-T5, enhances autonomous web automation by effectively handling open domains, long contexts, and HTML bias, resulting in a significant improvement in success rate on real-world websites.

<details>
<summary>Details</summary>
- **Abstract**: Pre-trained large language models (LLMs) have recently achieved better generalization and sample efficiency in autonomous web automation. However, the performance on real-world websites has still suffered from (1) open domainness, (2) limited context length, and (3) lack of inductive bias on HTML. We introduce WebAgent, an LLM-driven agent that learns from self-experience to complete tasks on real websites following natural language instructions. WebAgent plans ahead by decomposing instructions into canonical sub-instructions, summarizes long HTML documents into task-relevant snippets, and acts on websites via Python programs generated from those. We design WebAgent with Flan-U-PaLM, for grounded code generation, and HTML-T5, new pre-trained LLMs for long HTML documents using local and global attention mechanisms and a mixture of long-span denoising objectives, for planning and summarization. We empirically demonstrate that our modular recipe improves the success on real websites by over 50%, and that HTML-T5 is the best model to solve various HTML understanding tasks; achieving 18.7% higher success rate than the prior method on MiniWoB web automation benchmark, and SoTA performance on Mind2Web, an offline task planning evaluation.
- **OpenReview**: https://openreview.net/pdf?id=9JQtrumvg8
        
</details>

### MUSTARD: Mastering Uniform Synthesis of Theorem and Proof Data
> MUSTARD, a data generation framework, aims to address the challenge of limited training data for mathematical reasoning and theorem proving tasks in large language models (LLMs). It seamlessly synthesizes high-quality and diverse theorem and proof datasets, fostering the exploration of LLMs' reasoning abilities.

<details>
<summary>Details</summary>
- **Abstract**: Recent large language models (LLMs) have witnessed significant advancement in various tasks, including mathematical reasoning and theorem proving. As these two tasks require strict and formal multi-step inference, they are appealing domains for exploring the reasoning ability of LLMs but still formidable challenges. Previous studies such as Chain-of-Thought (CoT) have revealed the effectiveness of intermediate steps guidance. However, such step-wise annotation requires heavy labor, leading to insufficient training steps for current benchmarks. To fill this gap, this work introduces MUSTARD, a data generation framework that masters uniform synthesis of theorem and proof data of high quality and diversity. MUSTARD synthesizes data in three stages: (1) It samples a few mathematical concept seeds as the problem category. (2) Then it prompts a generative language model with the sampled concepts to obtain both the problems and their step-wise formal solutions. (3) Lastly, the framework utilizes a proof assistant (e.g., Lean Prover) to filter the valid proofs. With the proposed MUSTARD, we present a theorem-and-proof benchmark MUSTARDSAUCE with 7,335 valid data points. Each data point contains an informal statement, an informal proof, and a translated formal proof that passes the prover validation. We perform extensive analysis and demonstrate that MUSTARD generates validated high-quality step-by-step data. We further apply the MUSTARDSAUCE for fine-tuning smaller language models. The fine-tuned Llama 2-7B achieves improvements by 20.9% on zero-shot inference on GSM8K and achieves 8.7 of pass@1 on mathlib, which again demonstrates the data quality and their effectiveness on mathematical tasks.
- **OpenReview**: https://openreview.net/pdf?id=8xliOUg9EW
        
</details>

### Prometheus: Inducing Evaluation Capability in Language Models
> By leveraging reference materials and GPT-4's feedback training data, researchers have developed PROMETHEUS, an open-source evaluation-specific LLM that can assess long-form text with capabilities comparable to GPT-4, enabling practitioners to overcome the limitations of using closed-source models for large-scale and custom evaluation tasks.

<details>
<summary>Details</summary>
- **Abstract**: Recently, GPT-4 has become the de facto evaluator for long-form text generated by large language models (LLMs). However, for practitioners and researchers with large and custom evaluation tasks, GPT-4 is unreliable due to its closed-source nature, uncontrolled versioning, and prohibitive costs. In this work, we propose PROMETHEUS a fully open-source LLM that is on par with GPT-4s evaluation capabilities when the appropriate reference materials (reference answer, score rubric) are accompanied. For this purpose, we construct a new dataset  FEEDBACK COLLECTION  that consists of 1K fine-grained score rubrics, 20K instructions, and 100K natural language feedback generated by GPT-4. Using the FEEDBACK COLLECTION, we train PROMETHEUS, a 13B evaluation-specific LLM that can assess any given response based on novel and unseen score rubrics and reference materials provided by the user. Our datasets versatility and diversity make our model generalize to challenging real-world criteria, such as prioritizing conciseness, child-readability, or varying levels of formality. We show that PROMETHEUS shows a stronger correlation with GPT-4 evaluation compared to ChatGPT on seven evaluation benchmarks (Two Feedback Collection testsets, MT Bench, Vicuna Bench, Flask Eval, MT Bench Human Judgment, and HHH Alignment), showing the efficacy of our model and dataset design. During human evaluation with hand-crafted score rubrics, PROMETHEUS shows a Pearson correlation of 0.897 with human evaluators, which is on par with GPT-4-0613 (0.882), and greatly outperforms ChatGPT (0.392). Remarkably, when assessing the quality of the generated feedback, PROMETHEUS demonstrates a win rate of 58.62% when compared to GPT-4 evaluation and a win rate of 79.57% when compared to ChatGPT evaluation. Our findings suggests that by adding reference materials and training on GPT-4 feedback, we can obtain effective open-source evaluator LMs.
- **OpenReview**: https://openreview.net/pdf?id=8euJaTveKw
        
</details>

### OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models
> OmniQuant offers an innovative approach for calibrating quantization parameters in LLMs to optimize their performance under various quantization settings. Its unique components, LWC (Learnable Weight Clipping) and LET (Learnable Equivalent Transformation), efficiently address challenges in extreme weight values and activation outliers, respectively.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) have revolutionized natural language processing tasks. However, their practical deployment is hindered by their immense memory and computation requirements. Although recent post-training quantization (PTQ) methods are effective in reducing memory footprint and improving the computational efficiency of LLM, they hand-craft quantization parameters, which leads to low performance and fails to deal with extremely low-bit quantization. To tackle this issue, we introduce an Omnidirectionally calibrated Quantization (OmniQuant) technique for LLMs, which achieves good performance in diverse quantization settings while maintaining the computational efficiency of PTQ by efficiently optimizing various quantization parameters. OmniQuant comprises two innovative components including Learnable Weight Clipping (LWC) and Learnable Equivalent Transformation (LET). LWC modulates the extreme values of weights by optimizing the clipping threshold. Meanwhile, LET tackles activation outliers by shifting the challenge of quantization from activations to weights through a learnable equivalent transformation. Operating within a differentiable framework using block-wise error minimization, OmniQuant can optimize the quantization process efficiently for both weight-only and weight-activation quantization. For instance, the LLaMA-2 model family with the size of 7-70B can be processed with OmniQuant on a single A100-40G GPU within 1-16 hours using 128 samples. Extensive experiments validate OmniQuant's superior performance across diverse quantization configurations such as W4A4, W6A6, W4A16, W3A16, and W2A16. Additionally, OmniQuant demonstrates effectiveness in instruction-tuned models and delivers notable improvements in inference speed and memory reduction on real devices. Codes and models are available at https://github.com/anonymous998899/OmniQuant.
- **OpenReview**: https://openreview.net/pdf?id=8Wuvhh0LYW
        
</details>

### Context is Environment
> This study explores the connection between in-context learning and domain generalization, suggesting that considering context as environment could improve generalization performance. The proposed In-Context Risk Minimization algorithm demonstrates the benefits of this approach, highlighting the potential of in-context learning for both domain generalization and large language model research.

<details>
<summary>Details</summary>
- **Abstract**: Two lines of work are taking the central stage in AI research. On the one hand, the community is making increasing efforts to build models that discard spurious correlations and generalize better in novel test environments. Unfortunately, the hard lesson so far is that no proposal convincingly outperforms a simple empirical risk minimization baseline. On the other hand, large language models (LLMs) have erupted as algorithms able to learn in-context, generalizing on-the-fly to eclectic contextual circumstances that users enforce by means of prompting. In this paper, we argue that context is environment, and posit that in-context learning holds the key to better domain generalization. Via extensive theory and experiments, we show that paying attention to context$\\unicode{x2013}\\unicode{x2013}$unlabeled examples as they arrive$\\unicode{x2013}\\unicode{x2013}$allows our proposed In-Context Risk Minimization (ICRM) algorithm to zoom-in on the test environment risk minimizer, leading to significant out-of-distribution performance improvements. From all of this, two messages are worth taking home. Researchers in domain generalization should consider environment as context, and harness the adaptive power of in-context learning. Researchers in LLMs should consider context as environment, to better structure data towards generalization.
- **OpenReview**: https://openreview.net/pdf?id=8VPWfqtQMX
        
</details>

### Free from Bellman Completeness: Trajectory Stitching via Model-based Return-conditioned Supervised Learning
> Off-policy deep reinforcement learning algorithms utilizing return-conditioned supervised learning (RCSL) overcome the limitations of traditional Q-learning, converging reliably even with function approximation under relaxed assumptions. Notably, RCSL requires a constant network width for convergence, while Q-learning necessitates a width that grows linearly with the state space size.

<details>
<summary>Details</summary>
- **Abstract**: Off-policy dynamic programming (DP) techniques that implement fixed-point iteration, such as $Q$-learning, have proven to be an important technique for solving sequential decision-making problems. However, in the presence of function approximation such algorithms are not guaranteed to converge, often diverging due to the absence of Bellman-completeness in the function classes considered, a crucial condition for the success of DP-based methods.  In this paper, we show how off-policy learning techniques based on return-conditioned supervised learning (RCSL) are able to circumvent these challenges of Bellman completeness, converging unde significantly more relaxed assumptions inherited from supervised learning. We prove there exists a natural environment in which if one uses two-layer multilayer perceptron as the function approximator, the layer width needs to grow *linearly* with the state space size to satisfy Bellman-completeness while a constant layer width is enough for RCSL. These findings take a step towards explaining the superior empirical performance of RCSL methods compared to DP-based methods in many near-deterministic environments in deep reinforcement learning.  Furthermore, in order to learn from sub-optimal datasets, we propose a simple framework called MBRCSL, granting RCSL methods the ability of dynamic programming to stitch together segments from distinct trajectories. MBRCSL leverages learned dynamics models and forward sampling to accomplish trajectory stitching while avoiding the need for Bellman completeness that plagues all dynamic programming algorithms. We propose both theoretical analysis and experimental evaluation to back these claims, outperforming state-of-the-art model-free and model-based offline RL algorithms across several simulated robotics problems.
- **OpenReview**: https://openreview.net/pdf?id=7zY781bMDO
        
</details>

### BooookScore: A systematic exploration of book-length summarization in the era of LLMs
> This study explores how well large language models can summarize lengthy books by examining the coherence of their summaries. The results indicate that closed-source models perform better than open-source models, and while hierarchical merging produces more readable summaries, incremental updating captures more details, but with lower coherence.

<details>
<summary>Details</summary>
- **Abstract**: Summarizing book-length documents ($>$100K tokens)  that exceed the context window size of large language models (LLMs) requires first breaking the input document into smaller chunks and then prompting an LLM to merge, update, and compress chunk-level summaries. Despite the complexity and importance of this task, it has yet to be meaningfully studied due to the challenges of evaluation: existing book-length summarization datasets (e.g., BookSum) are in the pretraining data of most public LLMs, and existing evaluation methods struggle to capture errors made by modern LLM summarizers. In this paper, we present the first study of the coherence of LLM-based book-length summarizers implemented via two prompting workflows: (1) hierarchically merging chunk-level summaries, and (2) incrementally updating a running summary. We obtain 1193 fine-grained human annotations on GPT-4 generated summaries of 100 recently-published books and identify eight common types of coherence errors made by LLMs. Because human evaluation is expensive and time-consuming, we develop an automatic metric, BooookScore, that measures the proportion of sentences in a summary that do not contain any of the identified error types. BooookScore has high agreement with human annotations and allows us to systematically evaluate the impact of many other critical parameters (e.g., chunk size, base LLM) while saving \\$15K and 500 hours in human evaluation costs. We find that closed-source LLMs such as GPT-4 and Claude 2 produce summaries with higher BooookScore than the oft-repetitive ones generated by LLaMA 2. Incremental updating yields lower BooookScore but higher level of detail than hierarchical merging, a trade-off sometimes preferred by human annotators. We release code and annotations after blind review to spur more principled research on book-length summarization.
- **OpenReview**: https://openreview.net/pdf?id=7Ttk3RzDeu
        
</details>

### Generating Stealthy Jailbreak Prompts on Aligned Large Language Models
> AutoDAN, an automated jailbreak attack algorithm, generates stealthy prompts that bypass perplexity-based defense methods, uncovering the limitations of aligned LLMs and guiding their further development.

<details>
<summary>Details</summary>
- **Abstract**: The aligned Large Language Models (LLMs) are powerful language understanding and decision-making tools that are created through extensive alignment with human feedback. However, these large models remain susceptible to jailbreak attacks, where adversaries manipulate prompts to elicit malicious outputs that should not be given by aligned LLMs. Investigating jailbreak prompts can lead us to delve into the limitations of LLMs and further guide us to secure them. Unfortunately, existing jailbreak techniques suffer from either (1) scalability issues, where attacks heavily rely on manual crafting of prompts, or (2) stealthiness problems, as attacks depend on token-based algorithms to generate prompts that are often semantically meaningless, making them susceptible to detection through basic perplexity testing. In light of these challenges, we intend to answer this question: Can we develop an approach that can automatically generate stealthy jailbreak prompts? In this paper, we introduce AutoDAN, a novel jailbreak attack against aligned LLMs. AutoDAN can automatically generate stealthy jailbreak prompts by the carefully designed hierarchical genetic algorithm. Extensive evaluations demonstrate that AutoDAN not only automates the process while preserving semantic meaningfulness, but also demonstrates superior attack strength in cross-model transferability, and cross-sample universality compared with the baseline. Moreover, we also compare AutoDAN with perplexity-based defense methods and show that AutoDAN can bypass them effectively.
- **OpenReview**: https://openreview.net/pdf?id=7Jwpw4qKkb
        
</details>

### A Semantic Invariant Robust Watermark for Large Language Models
> Investigating the trade-off between attack and security robustness in LLM watermarking, this study presents a novel semantic invariant watermarking method for LLMs that leverages semantic embeddings to achieve both robustness aspects, as demonstrated through theoretical analyses and experimental evaluations.

<details>
<summary>Details</summary>
- **Abstract**: Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subsequent analyses and experiments demonstrated the attack robustness of our method in semantically invariant settings: synonym substitution and text paraphrasing settings. Finally, we also show that our watermark possesses adequate security robustness.
- **OpenReview**: https://openreview.net/pdf?id=6p8lpe4MNf
        
</details>

### Large Language Model Cascades with Mixture of Thought Representations for Cost-Efficient Reasoning
> This study aims to create an LLM cascade that reduces the cost of using more powerful LLMs by employing a weaker LLM to handle simpler tasks while leveraging consistency checks to delegate complex questions to the stronger model, potentially saving 60% of costs while achieving comparable performance.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) such as GPT-4 have exhibited remarkable performance in a variety of tasks, but this strong performance often comes with the high expense of using paid API services. In this paper, we are motivated to study building an LLM "cascade" to save the cost of using LLMs, particularly for performing (e.g., mathematical, causal) reasoning tasks. Our cascade pipeline follows the intuition that simpler questions can be addressed by a weaker but more affordable LLM, whereas only the most challenging questions necessitate the stronger and more expensive LLM. To realize this decision-making, we consider the "answer consistency" of the weaker LLM as a signal of the question difficulty and propose several methods for answering sampling and consistency checking, including one leveraging a mixture of two thought representations (i.e., Chain-of-Thought and Program-of-Thought). Through experiments on six reasoning benchmark datasets, with GPT-3.5-turbo and GPT-4 being the weaker and stronger LLMs, respectively, our cascade pipeline demonstrates comparable performance but reduces about 60% of the cost compared with fully using the stronger LLM.
- **OpenReview**: https://openreview.net/pdf?id=6okaSfANzh
        
</details>

### Mixture-of-Experts Meets Instruction Tuning: A Winning Combination for Large Language Models
> Combining Sparse Mixture-of-Experts (MoE) with instruction tuning enhances the performance of Large Language Models (LLMs) on downstream tasks, demonstrating that MoE models benefit from instruction tuning more than dense models. This finding suggests a reevaluation of design principles for large-scale, high-performance language models in the context of task-agnostic learning.

<details>
<summary>Details</summary>
- **Abstract**: Sparse Mixture-of-Experts (MoE) is a neural architecture design that can be utilized to add learnable parameters to Large Language Models (LLMs) without increasing inference cost. Instruction tuning is a technique for training LLMs to follow instructions. We advocate combining these two approaches, as we find that MoE models benefit more from instruction tuning than dense models. In particular, we conduct empirical studies across three experimental setups: (i) Direct finetuning on individual downstream tasks devoid of instruction tuning; (ii) Instruction tuning followed by in-context few-shot or zero-shot generalization on downstream tasks; and (iii) Instruction tuning supplemented by further finetuning on individual downstream tasks. In the first scenario, MoE models overall underperform dense models of identical computational capacity. This narrative, however, dramatically changes with the introduction of instruction tuning (second and third scenario), used independently or in conjunction with task-specific finetuning. Our most powerful model, FLAN-MOE32B, surpasses the performance of FLAN-PALM62B on four benchmark tasks, while using only a third of the FLOPs. The advancements embodied by FLAN-MOE inspire a reevaluation of the design principles of large-scale, high-performance language models in the framework of task-agnostic learning.
- **OpenReview**: https://openreview.net/pdf?id=6mLjDwYte5
        
</details>

### LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models
> LongLoRA significantly reduces the training cost of extending context sizes in LLMs by combining sparse local attention in fine-tuning and parameter-efficient fine-tuning, enabling models to process longer sequences with comparable performance and compatibility with existing techniques.

<details>
<summary>Details</summary>
- **Abstract**: We present LongLoRA, an efficient fine-tuning approach that extends the context sizes of pre-trained large language models (LLMs), with limited computation cost. Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. For example, training on the context length of 8192 needs 16x computational costs in self-attention layers as that of 2048. In this paper, we speed up the context extension of LLMs in two aspects. On the one hand, although dense global attention is needed during inference, fine-tuning the model can be effectively and efficiently done by sparse local attention. The proposed shift short attention effectively enables context extension, leading to non-trivial computation saving with similar performance to fine-tuning with vanilla attention. Particularly, it can be implemented with only two lines of code in training, while being optional in inference. On the other hand, we revisit the parameter-efficient fine-tuning regime for context expansion. Notably, we find that LoRA for context extension works well under the premise of trainable embedding and normalization. LongLoRA demonstrates strong empirical results on various tasks on Llama2 models from 7B/13B to 70B. LongLoRA adopts Llama2 7B from 4k context to 100k, or Llama2 70B to 32k on a single 8$\\times$ A100 machine. LongLoRA extends models' context while retaining their original architectures, and is compatible with most existing techniques, like Flash-Attention2. In addition, we further conduct supervised fine-tuning on our LongLoRA models, with long instruction-following data. Our code and models will be publicly available.
- **OpenReview**: https://openreview.net/pdf?id=6PmJoRfdaK
        
</details>

### In-Context Learning Dynamics with Random Binary Sequences
> This paper presents a novel framework for interpreting the complex capabilities of large language models (LLMs) by studying how they learn from random binary sequences, revealing their emergent abilities to generate pseudo-random numbers and learn formal languages, with sharp transitions in their learning dynamics.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) trained on huge corpora of text datasets demonstrate complex, emergent capabilities, achieving state-of-the-art performance on tasks they were not explicitly trained for. The precise nature of LLM capabilities is often unclear, and different prompts can elicit different capabilities through in-context learning. We propose a Cognitive Interpretability framework that enables us to analyze in-context learning dynamics to understand latent concepts in LLMs underlying behavioral patterns. This provides a more nuanced understanding than success-or-failure evaluation benchmarks, but does not require observing internal activations as a mechanistic interpretation of circuits would require. Inspired by the cognitive science of human randomness perception, we use random binary sequences as context and study dynamics of in-context learning by manipulating properties of context data, such as sequence length. In the latest GPT-3.5+ models, we find emergent abilities to generate pseudo-random numbers and learn basic formal languages, with striking in-context learning dynamics where model outputs transition sharply from pseudo-random behaviors to deterministic repetition.
- **OpenReview**: https://openreview.net/pdf?id=62K7mALO2q
        
</details>

### MiniLLM: Knowledge Distillation of Large Language Models
> This work introduces MiniLLM, a new approach for effectively distilling smaller language models from larger, white-box generative language models. By deploying reverse Kullback-Leibler divergence as the optimization objective, MiniLLM ensures that the student model captures the core knowledge of the teacher model while minimizing overestimation of low-probability regions.

<details>
<summary>Details</summary>
- **Abstract**: Knowledge Distillation (KD) is a promising technique for reducing the high computational demand of large language models (LLMs). However, previous KD methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge from white-box generative LLMs is still under-explored, which becomes more and more important with the prosperity of LLMs. In this work, we propose MiniLLM that distills smaller language models from generative larger language models. We first replace the forward Kullback-Leibler divergence (KLD) objective in the standard KD approaches with reverse KLD, which is more suitable for KD on generative language models, to prevent the student model from overestimating the low-probability regions of the teacher distribution. Then, we derive an effective optimization approach to learn this objective. Extensive experiments in the instruction-following setting show that the MiniLLM models generate more precise responses with the higher overall quality, lower exposure bias, better calibration, and higher long-text generation performance. Our method is also scalable for different model families with 120M to 13B parameters. Our code can be found in the supplementary material.
- **OpenReview**: https://openreview.net/pdf?id=5h0qf7IBZZ
        
</details>

### PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization
> PandaLM, a large language model (LLM) evaluator, can reliably judge the quality of LLMs, considering both objective and subjective factors without relying on external data or costly human evaluations, leading to fairer and more effective LLM tuning.

<details>
<summary>Details</summary>
- **Abstract**: Instruction tuning large language models (LLMs) remains a challenging task, owing to the complexity of hyperparameter selection and the difficulty involved in evaluating the tuned models. To determine the optimal hyperparameters, an automatic, robust, and reliable evaluation benchmark is essential. However, establishing such a benchmark is not a trivial task due to the challenges associated with evaluation accuracy and privacy protection. In response to these challenges, we introduce a judge large language model, named PandaLM, which is trained to distinguish the superior model given several LLMs. PandaLM's focus extends beyond just the objective correctness of responses, which is the main focus of traditional evaluation datasets. It addresses vital subjective factors such as relative conciseness, clarity, adherence to instructions, comprehensiveness, and formality. To ensure the reliability of PandaLM, we collect a diverse human-annotated test dataset, where all contexts are generated by humans and labels are aligned with human preferences. Our findings reveal that PandaLM-7B offers a performance comparable to both GPT-3.5 and GPT-4. Impressively, PandaLM-70B surpasses their performance. PandaLM enables the evaluation of LLM to be fairer but with less cost, evidenced by significant improvements achieved by models tuned through PandaLM compared to their counterparts trained with default Alpaca's hyperparameters. In addition, PandaLM does not depend on API-based evaluations, thus avoiding potential data leakage.
- **OpenReview**: https://openreview.net/pdf?id=5Nn2BLV7SB
        
</details>

### Multiscale Positive-Unlabeled Detection of AI-Generated Texts
> Long-text detection by language models is augmented by a novel training framework (MPU) that addresses the challenge of short-text detection by incorporating these texts as partially unlabeled entities, resulting in improved detection performance on both short and long AI-generated texts.

<details>
<summary>Details</summary>
- **Abstract**: Recent releases of Large Language Models (LLMs), e.g. ChatGPT, are astonishing at generating human-like texts, but they may impact the authenticity of texts. Previous works proposed methods to detect these AI-generated texts, including simple ML classifiers, pretrained-model-based zero-shot methods, and finetuned language classification models. However, mainstream detectors always fail on short texts, like SMSes, Tweets, and reviews. In this paper, a Multiscale Positive-Unlabeled (MPU) training framework is proposed to address the difficulty of short-text detection without sacrificing long-texts. Firstly, we acknowledge the human-resemblance property of short machine texts, and rephrase AI text detection as a partial Positive-Unlabeled (PU) problem by regarding these short machine texts as partially "unlabeled". Then in this PU context, we propose the length-sensitive Multiscale PU Loss, where a recurrent model in abstraction is used to estimate positive priors of scale-variant corpora. Additionally, we introduce a Text Multiscaling module to enrich training corpora. Experiments show that our MPU method augments detection performance on long AI-generated texts, and significantly improves short-text detection of language model detectors. Language Models trained with MPU could outcompete existing detectors on various short-text and long-text detection benchmarks.
- **OpenReview**: https://openreview.net/pdf?id=5Lp6qU9hzV
        
</details>

### MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning
> This study presents a novel approach, MMICL, that enhances vision-language models' ability to comprehend complex, multimodal prompts by introducing a context scheme and dataset tailored for in-context learning. MMICL empowers VLMs to effectively tackle vision-language tasks, particularly those involving complex prompts, outperforming existing methods.

<details>
<summary>Details</summary>
- **Abstract**: Since the resurgence of deep learning, vision-language models (VLMs) enhanced by large language models (LLMs) have grown exponentially in popularity.  However, while LLMs can utilize extensive background knowledge and task information with in-context learning, most VLMs still struggle with understanding complex multi-modal prompts with multiple images, making VLMs less effective in downstream vision-language tasks. In this paper, we address the limitation above by 1) introducing MMICL, a new approach to allow the VLM to deal with multi-modal inputs efficiently; 2) proposing a novel context scheme to augment the in-context learning ability of the VLM; 3) constructing the Multi-modal In-Context Learning (MIC) dataset, designed to enhance the VLM's ability to understand complex multi-modal prompts. Our experiments confirm that MMICL achieves new state-of-the-art zero-shot performance on a wide range of general vision-language tasks, especially for complex benchmarks, including MME and MMBench. Our analysis demonstrates that MMICL effectively tackles the challenge of complex multi-modal prompt understanding and emerges the impressive ICL ability. Furthermore, we observe that MMICL successfully alleviates language bias in VLMs, a common issue for VLMs that often leads to hallucination when faced with extensive textual context. Our code, dataset and model are available at github link.
- **OpenReview**: https://openreview.net/pdf?id=5KojubHBr8
        
</details>

### When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method
> This paper delves into the impact of scaling factors on finetuning performance of large language models, revealing that scaling the LLM model size boosts performance more than other factors and that the optimal finetuning method varies depending on the task and dataset.

<details>
<summary>Details</summary>
- **Abstract**: While large language models (LLMs) often adopt finetuning to unlock their capabilities for downstream applications, our understanding on the inductive biases (especially the scaling properties) of different finetuning methods is still limited. To fill this gap, we conduct systematic experiments studying whether and how different scaling factors, including LLM model size, pretraining data size, new finetuning parameter size and finetuning data size, affect the finetuning performance. We consider two types of finetuning  full-model tuning (FMT) and parameter efficient tuning (PET, including prompt tuning and LoRA), and explore their scaling behaviors in the data-limited regime where the LLM model size substantially outweighs the finetuning data size. Based on two sets of pretrained bilingual LLMs from 1B to 16B and experiments on bilingual machine translation and multilingual summarization benchmarks, we find that 1) LLM finetuning follows a powerbased multiplicative joint scaling law between finetuning data size and each other scaling factor; 2) LLM finetuning benefits more from LLM model scaling than pretraining data scaling, and PET parameter scaling is generally ineffective; and 3) the optimal finetuning method is highly task- and finetuning data-dependent. We hope our findings could shed light on understanding, selecting and developing LLM finetuning methods.
- **OpenReview**: https://openreview.net/pdf?id=5HCnKDeTws
        
</details>

### How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions
> An exploration into lie detection for Large Language Models (LLMs) reveals that LLMs exhibit consistent behavioral patterns when lying, regardless of architecture or context. This suggests the feasibility of developing general-purpose lie detectors that can identify lies even without direct access to LLM activations or ground-truth knowledge.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) can lie, which we define as outputting false statements despite knowing the truth in a demonstrable sense. An example is an LLM instructed to spread misinformation. Here, we conduct an initial exploration into the feasibility of lie detection for LLMs. We develop a simple lie detector that requires neither access to the LLMs activations (black-box) nor ground-truth knowledge of the fact in question. The detector works by asking a predefined set of unrelated follow-up questions after a suspected lie, and feeding the LLMs yes/no answers into a logistic regression classifier. Despite its simplicity, this lie detector is highly accurate and surprisingly general. When trained on examples from a single settingprompting GPT-3.5 to lie about factual questionsthe detector generalises out-of-distribution to (1) other LLM architectures, (2) LLMs fine-tuned to lie, (3) sycophantic lies, and (4) lies emerging in real-life scenarios such as sales. These results indicate that LLMs have distinctive lie-related behavioural patterns, consistent across architectures and contexts, which could enable general-purpose lie detection'
- **OpenReview**: https://openreview.net/pdf?id=567BjxgaTp
        
</details>

### Language Model Detectors Are Easily Optimized Against
> This study reveals a highly effective way to trick AI detectors designed to spot text written by language models. By slightly altering a language model through reinforcement learning, the researchers demonstrate that its output can confuse these detectors while retaining its fluency and broad applicability.

<details>
<summary>Details</summary>
- **Abstract**: The fluency and general applicability of large language models (LLMs) has motivated significant interest in detecting whether a piece of text was written by a language model. While both academic and commercial detectors have been deployed in some settings, particularly education, other research has highlighted the fragility of these systems. In this paper, we demonstrate a data-efficient attack that fine-tunes language models to confuse existing detectors, leveraging recent developments in reinforcement learning of language models. We use the 'human-ness' score (often just a log probability) of various open-source and commercial detectors as a reward function for reinforcement learning, subject to a KL-divergence constraint that the resulting model does not differ significantly from the original. For a 7B parameter Llama-2 model, fine-tuning for under a day reduces the AUROC of the OpenAI RoBERTa-Large detector from 0.84 to 0.62, while perplexity on OpenWebText increases from 8.7 to only 9.0; with a larger perplexity budget, we reduce AUROC to 0.30 (worse than random), with a perplexity increase to 9.9. Similar to traditional adversarial attacks, we find that this increase in 'detector evasion' generalizes to other detectors not used during training. In light of our empirical results, we advise against continued reliance on LLM-generated text detectors.
- **OpenReview**: https://openreview.net/pdf?id=4eJDMjYZZG
        
</details>

### Chain-of-Table: Evolving Tables in the Reasoning Chain for Table Understanding
> In this paper, the authors introduce Chain-of-Table, a framework for reasoning over tabular data using large language models. By leveraging tabular data as intermediate thoughts, Chain-of-Table enables LLMs to dynamically plan reasoning steps, leading to improved accuracy and reliability in table-based reasoning tasks.

<details>
<summary>Details</summary>
- **Abstract**: Table-based reasoning with large language models (LLMs) is a promising direction to tackle many table understanding tasks, such as table-based question answering and fact verification. Compared with generic reasoning, table-based reasoning requires the extraction of underlying semantics from both free-form questions and semi-structured tabular data. Chain-of-Thought and its similar approaches incorporate the reasoning chain in the form of textual context, but it is still an open question how to effectively leverage tabular data in the reasoning chain. We propose the Chain-of-Table framework, where tabular data is explicitly used in the reasoning chain as a proxy for intermediate thoughts. Specifically, we guide LLMs using in-context samples to iteratively generate operations and update the table to represent a complex reasoning chain. LLMs can therefore dynamically plan the next operation based on the results of the previous ones. This continuous evolution of the table forms a chain, showing the reasoning process for a given tabular problem. The chain carries structured information of the intermediate results, enabling more accurate and reliable predictions. Chain-of-Table achieves new state-of-the-art performance on WikiTQ, FeTaQA, and TabFact benchmarks across multiple LLM choices.
- **OpenReview**: https://openreview.net/pdf?id=4L0xnS4GQM
        
</details>

### Curiosity-driven Red-teaming for Large Language Models
> Curiosity-driven exploration, an innovative method, enhances the diversity and effectiveness of "red teaming" LLMs, which automatically probe when larger LLMs generate inappropriate content, making it a promising tool for evaluating and mitigating language model risks.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) hold great potential for various natural language applications but risk generating incorrect or toxic content. In order to probe when an LLM generates unwanted content, the current paradigm is to recruit human testers to create input prompts (i.e., test cases) designed to elicit unfavorable responses from LLMs. This procedure is called red teaming. However, relying solely on human testers can be both expensive and time-consuming. Recent works automate red teaming by training LLMs (i.e., red team LLMs) with reinforcement learning (RL) to maximize the chance of eliciting undesirable responses (i.e., successful test cases) from the target LLMs being evaluated. However, while effective at provoking undesired responses, current RL methods lack test case diversity as RL-based methods tend to consistently generate the same few successful test cases once found. To overcome this limitation, we introduce curiosity-driven exploration to train red team models. This approach jointly maximizes the test case effectiveness and novelty. Maximizing novelty motivates the red-team model to search for new and diverse test cases. We evaluate our method by performing red teaming against LLMs in text continuation and instruction following tasks. Our experiments show that curiosity-driven exploration achieves greater diversity in all the experiments compared to existing RL-based red team methods while maintaining effectiveness. Remarkably, curiosity-driven exploration also enhances the effectiveness when performing red teaming in instruction following test cases, generating a higher number of successful test cases. We even demonstrate that curiosity-driven exploration successfully provokes toxic responses from the LLaMA2 model that has undergone finetuning based on human preferences.
- **OpenReview**: https://openreview.net/pdf?id=4KqkizXgXU
        
</details>

### Detecting Machine-Generated Texts by Multi-Population Aware Optimization for Maximum Mean Discrepancy
> In this work, the authors propose MMD-MP, a method that addresses the challenges of detecting machine-generated text (MGT) by exploiting the maximum mean discrepancy (MMD) while accounting for the issue of multiple text populations in diverse MGTs. This approach helps stabilize MMD and enables improved detection accuracy, outperforming other methods in paragraph-based and sentence-based detection tasks.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) such as ChatGPT have exhibited remarkable performance in generating human-like texts. However, machine-generated texts (MGTs) may carry critical risks, such as plagiarism issues and hallucination information. Therefore, it is very urgent and important to detect MGTs in many situations. Unfortunately, it is challenging to distinguish MGTs and human-written texts because the distributional discrepancy between them is often very subtle due to the remarkable performance of LLMS. In this paper, we seek to exploit \\textit{maximum mean discrepancy} (MMD) to address this issue in the sense that MMD can well identify distributional discrepancies. However,  directly training a detector with MMD using diverse MGTs will incur a significantly increased variance of MMD since MGTs may contain \\textit{multiple text populations} due to various LLMs. This will severely impair MMD's ability to measure the difference between two samples. To tackle this, we propose a novel \\textit{multi-population} aware optimization method for MMD called MMD-MP, which can \\textit{avoid variance increases} and thus improve the stability to measure the distributional discrepancy. Relying on MMD-MP, we develop two methods for paragraph-based and sentence-based detection, respectively. Extensive experiments on various LLMs, \\eg,  GPT2 and ChatGPT, show superior detection performance of our MMD-MP.
- **OpenReview**: https://openreview.net/pdf?id=3fEKavFsnv
        
</details>

### LEGO-Prover: Neural Theorem Proving with Growing Libraries
> LEGO-Prover, an LLM-based theorem prover, introduces a growing library of proven theorems and lemmas, fostering modularity and reusability. This enables advancements in the field of theorem proving, particularly by enabling the creation of new theorems and bridging the gap between human and formal proofs.

<details>
<summary>Details</summary>
- **Abstract**: Despite the success of large language models (LLMs), the task of theorem proving still remains one of the hardest reasoning tasks that is far from being fully solved. Prior methods using language models have demonstrated promising results, but they still struggle to prove even middle school level theorems. One common limitation of these methods is that they assume a fixed theorem library during the whole theorem proving process. However, as we all know, creating new useful theorems or even new theories is not only helpful but crucial and necessary for advancing mathematics and proving harder and deeper results. In this work, we present LEGO-Prover, which employs a growing skill library containing verified lemmas as skills to augment the capability of LLMs used in theorem proving. By constructing the proof modularly, LEGO-Prover enables LLMs to utilize existing skills retrieved from the library and to create new skills during the proving process. These skills are further evolved (by prompting an LLM) to enrich the library on another scale. Modular and reusable skills are constantly added to the library to enable tackling increasingly intricate mathematical problems. Moreover, the learned library further bridges the gap between human proofs and formal proofs by making it easier to impute missing steps. LEGO-Prover advances the state-of-the-art pass rate on miniF2F-valid (48.0% to 57.0%) and miniF2F-test (45.5% to 47.1%). During the proving process, LEGO-Prover also manages to generate over 20,000 skills (theorems/lemmas) and adds them to the growing library. Our ablation study indicates that these newly added skills are indeed helpful for proving theorems, resulting in an improvement from a success rate of 47.1% to 50.4%. We also release our code and all the generated skills.
- **OpenReview**: https://openreview.net/pdf?id=3f5PALef5B
        
</details>

### Privately Aligning Language Models with Reinforcement Learning
> This paper explores the privacy-preserving alignment of Large Language Models (LLMs) through Reinforcement Learning (RL) and Differential Privacy (DP). The study examines two alignment paradigms: RL without human intervention and RL from human feedback. DP techniques are incorporated to ensure user privacy, balancing the trade-off between data utility and privacy protection.

<details>
<summary>Details</summary>
- **Abstract**: Positioned between pre-training and user deployment, aligning large language models (LLMs) through reinforcement learning (RL) has emerged as a prevailing strategy for training instruction following-models such as ChatGPT. In this work, we initiate the study of privacy-preserving alignment of LLMs through Differential Privacy (DP) in conjunction with RL. Following the influential work of Ziegler et al. (2020), we study two dominant paradigms: (i) alignment via RL without human in the loop (e.g., positive review generation) and (ii) alignment via RL from human feedback (RLHF) (e.g., summarization in a human-preferred way). We give a new DP framework to achieve alignment via RL, and prove its correctness. Our experimental results validate the effectiveness of our approach, offering competitive utility while ensuring strong privacy protections.
- **OpenReview**: https://openreview.net/pdf?id=3d0OmYTNui
        
</details>

### Step-Back Prompting Enables Reasoning Via Abstraction in Large Language Models
> Step-Back Prompting elevates LLMs' reasoning abilities by guiding them towards overarching principles and concepts, leading to significant performance enhancements in STEM, knowledge questions, and multi-step reasoning tasks.

<details>
<summary>Details</summary>
- **Abstract**: We present Step-Back Prompting, a simple prompting technique that enables LLMs to do abstractions to derive high-level concepts and first principles from instances containing specific details. Using the concepts and principles to guide the reasoning steps, LLMs significantly improve their abilities in following a correct reasoning path towards the solution. We conduct experiments of Step-Back Prompting with PaLM-2 models and observe substantial performance gains on a wide range of challenging reasoning-intensive tasks including STEM, Knowledge QA, and Multi-Hop Reasoning. For instance, Step-Back Prompting improves PaLM-2L performance on MMLU Physics and Chemistry by 7% and 11%, TimeQA by 34%, and MuSiQue by 7%.
- **OpenReview**: https://openreview.net/pdf?id=3bq3jsvcQ1
        
</details>

### PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training
> PoSE training, an innovative approach, enables Large Language Models to handle long inputs efficiently by simulating them within a fixed context window. Despite its memory and time-saving benefits compared to traditional methods, PoSE maintains model performance, opening new possibilities for extending context window length and enabling potential infinite length handling.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) are trained with a pre-defined context length, restricting their use in scenarios requiring long inputs. Previous efforts for adapting LLMs to a longer length usually requires fine-tuning with this target length (Full-length fine-tuning), suffering intensive training cost. To decouple train length from target length for efficient context window extension, we propose Positional Skip-wisE (PoSE) training that smartly simulates long inputs using a fixed context window. This is achieved by first dividing the original context window into several chunks, then designing distinct skipping bias terms to manipulate the position indices of each chunk. These bias terms and the lengths of each chunk are altered for every training example, allowing the model to adapt to all positions within target length. Experimental results show that PoSE greatly reduces memory and time overhead compared with Full-length fine-tuning, with minimal impact on performance. Leveraging this advantage, we have successfully extended the LLaMA model to 128k tokens using a 2k training context window. Furthermore, we empirically confirm that PoSE is compatible with all RoPE-based LLMs and position interpolation strategies. Notably, our method can potentially support infinite length, limited only by memory usage in inference. With ongoing progress for efficient inference, we believe PoSE can further scale the context window beyond 128k.
- **OpenReview**: https://openreview.net/pdf?id=3Z1gxuAQrA
        
</details>

### Chain of Thought Empowers Transformers to Solve Inherently Serial Problems
> By providing an intermediate sequence of steps, chain of thought (CoT) empowers transformers with the ability to perform serial computations, enhancing their expressiveness in solving arithmetic and symbolic reasoning problems.

<details>
<summary>Details</summary>
- **Abstract**: Generating a sequence of intermediate steps, \\emph{a.k.a.}, a chain of thought (CoT), is a highly effective method to improve the accuracy of large language models (LLMs) on arithmetics and symbolic reasoning tasks. However, the mechanism behind CoT remains unclear.  This work provides a theoretical understanding of the power of CoT for decoder-only transformers through the lens of expressiveness. Conceptually, CoT empowers the model with the ability to perform inherently serial computation, which is otherwise lacking in transformers, especially when depth is low. Given input length $n$, previous works have constant-depth transformers with finite precision $\\mathsf{poly}(n)$ embedding size can only solve problems in $\\mathsf{TC}^0$ without CoT. We first show an even tighter expressiveness upper bound for constant-depth transformers with constant-bit precision, which can only solve problems in $\\mathsf{AC}^0$, a proper subset of $ \\mathsf{TC}^0$. However, with $T$ steps of CoT, constant-depth transformers using constant-bit precision and $O(\\log n)$ embedding size can solve any problem solvable by boolean circuits of size $T$. Empirically, enabling CoT dramatically improves the accuracy for tasks that are hard for parallel computation, including the composition of permutation groups, iterated squaring, and circuit value problems, especially for low-depth transformers.
- **OpenReview**: https://openreview.net/pdf?id=3EWTEy9MTM
        
</details>

### Enable Lanuguage Models to Implicitly Learn Self-Improvement From Data
> Can Large Language Models improve their responses without explicit human guidance? This study introduces PIT, a framework that implicitly learns improvement goals from human preferences, reducing the need for extensive human annotation and demonstrating significantly better performance than prompting-based methods.

<details>
<summary>Details</summary>
- **Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in open-ended text generation tasks. However, the inherent open-ended nature of these tasks implies that there is always room for improvement in the quality of model responses. To address this challenge, various approaches have been proposed to enhance the performance of LLMs. There has been a growing focus on enabling LLMs to self-improve their response quality, thereby reducing the reliance on extensive human annotation efforts for collecting diverse and high-quality training data. Recently, prompting-based methods have been widely explored among self-improvement methods owing to their effectiveness, efficiency, and convenience. However, those methods usually require explicitly and thoroughly written rubrics as inputs to LLMs. It is expensive and challenging to manually derive and provide all necessary rubrics with a real-world complex goal for improvement (e.g., being more helpfulness and less harmful). To this end, we propose an imPlicit self-ImprovemenT (PIT) framework that implicitly learns the improvement goal from human preference data. PIT only requires preference data that are used to train reward models with no extra human efforts. Specifically, we reformulate the training objective of reinforcement learning from human feedback (RLHF) -- instead of maximizing response quality for a given input, we maximize the quality gap of the response conditioned on a reference response. In this way, PIT is implicitly trained with the improvement goal of better aligning with human preferences. Experiments on two real-world datasets and one synthetic dataset show that our method significantly outperforms prompting-based methods.
- **OpenReview**: https://openreview.net/pdf?id=2tVHNRZuCs
        
</details>

### Ferret: Refer and Ground Anything Anywhere at Any Granularity
> Ferret, a multimodal AI model, revolutionizes image referring and grounding by utilizing a unique hybrid region representation that seamlessly combines discrete coordinates with continuous features. This enables Ferret to understand and interpret spatial references of any shape or size within an image, making it adaptable to diverse region inputs and optimizing its multimodal capabilities in chat-based scenarios emphasizing region-based and localization-specific tasks.

<details>
<summary>Details</summary>
- **Abstract**: We introduce Ferret, a new Multimodal Large Language Model (MLLM) capable of understanding spatial referring of any shape or granularity within an image and accurately grounding open-vocabulary descriptions. To unify referring and grounding in the LLM paradigm, Ferret employs a novel and powerful hybrid region representation that integrates discrete coordinates and continuous features jointly to represent a region in the image. To extract the continuous features of versatile regions,  we propose a spatial-aware visual sampler, adept at handling varying sparsity across different shapes. Consequently, Ferret can accept diverse region inputs, such as points, bounding boxes, and free-form shapes. To bolster the desired capability of Ferret, we curate GRIT, a comprehensive refer-and-ground instruction tuning dataset including 1.1M samples that contain rich hierarchical spatial knowledge, with an additional 130K hard negative data to promote model robustness. The resulting model not only achieves superior performance in classical referring and grounding tasks, but also greatly outperforms existing MLLMs in region-based and localization-demanded multimodal chatting. Our evaluations also reveal a significantly improved capability of describing image details and a remarkable alleviation in object hallucination.
- **OpenReview**: https://openreview.net/pdf?id=2msbbX3ydD
        
</details>

### Beyond Reverse KL: Generalizing Direct Preference Optimization with Diverse Divergence Constraints
> This paper introduces $f$-DPO, a generalized approach to aligning large language models with human preferences through direct preference optimization. By incorporating diverse divergence constraints, $f$-DPO enables a more efficient and supervised mapping between reward functions and optimal policies, reducing the need for complex and computationally expensive reinforcement learning from human feedback while maintaining or even improving alignment performance.

<details>
<summary>Details</summary>
- **Abstract**: The increasing capabilities of large language models (LLMs) raise opportunities for artificial general intelligence but concurrently amplify safety concerns, such as potential misuse of AI systems, necessitating effective AI alignment. Reinforcement Learning from Human Feedback (RLHF) has emerged as a promising pathway towards AI alignment but brings forth challenges due to its complexity and dependence on a separate reward model. Direct Preference Optimization (DPO) has been proposed as an alternative; and it remains equivalent to RLHF under the reverse KL regularization constraint. This paper presents $f$-DPO, a generalized approach to DPO by incorporating diverse divergence constraints. We show that under certain $f$-divergences, including Jensen-Shannon divergence, forward KL divergences and $\\alpha$-divergences, the complex relationship between the reward and optimal policy can also be simplified by addressing the KarushKuhnTucker conditions. This eliminates the need for estimating the normalizing constant in the Bradley-Terry model and enables a tractable mapping between the reward function and the optimal policy. Our approach optimizes LLMs to align with human preferences in a more efficient and supervised manner under a broad set of divergence constraints. Empirically, adopting these divergences ensures a balance between alignment performance and generation diversity. Importantly, our $f$-DPO outperforms PPO-based methods in divergence efficiency, and divergence constraints directly influence expected calibration error (ECE).
- **OpenReview**: https://openreview.net/pdf?id=2cRzmWXK9N
        
</details>

### MEND: Meta Demonstration Distillation for Efficient and Effective In-Context Learning
> Researchers propose a novel method to efficiently leverage demonstrations in context learning tasks, reducing computational cost without compromising performance. By employing meta-distillation, their approach effectively and adaptively distills demonstrations into compact vectors, enhancing both efficiency and effectiveness in various in-context learning scenarios.

<details>
<summary>Details</summary>
- **Abstract**: Large Language models (LLMs) have demonstrated impressive in-context learning (ICL) capabilities,  where a LLM makes predictions for a given test input together with a few input-output pairs (demonstrations). Nevertheless, the inclusion of demonstrations poses a challenge, leading to a quadratic increase in the computational overhead of the self-attention mechanism. Existing solutions attempt to condense lengthy demonstrations into compact vectors.  However, they often require task-specific retraining or compromise LLM's in-context learning performance.  To mitigate these challenges, we present Meta Demonstration Distillation (MEND), where a language model learns to distill any lengthy demonstrations into vectors without retraining for a new downstream task.  We exploit the knowledge distillation to enhance alignment between MEND and MEND, achieving both efficiency and effectiveness concurrently.  MEND is endowed with the meta-knowledge of distilling demonstrations through a two-stage training process, which includes meta-distillation pretraining and fine-tuning. Comprehensive evaluations across seven diverse ICL settings using decoder-only (GPT-2) and encoder-decoder (T5) attest to MEND's prowess. It not only matches but often outperforms the Vanilla ICL as well as other state-of-the-art distillation models, while significantly reducing the computational demands.  This innovation promises enhanced scalability and efficiency for the practical deployment of large language models.
- **OpenReview**: https://openreview.net/pdf?id=2Y5kBPtU0o
        
</details>

### Time Travel in LLMs: Tracing Data Contamination in Large Language Models
> The research proposes a method to detect whether large language models (LLMs) are contaminated with test data from downstream tasks in their training data. The method identifies potential contamination at the instance level using guided prompts and then assesses wider contamination at the partition level using statistical methods or classifiers.

<details>
<summary>Details</summary>
- **Abstract**: Data contamination, i.e., the presence of test data from downstream tasks in the training data of large language models (LLMs), is a potential major issue in measuring LLMs\' real effectiveness on other tasks. We propose a straightforward yet effective method for identifying data contamination within LLMs. At its core, our approach starts by identifying potential contamination at the instance level; using this information, our approach then assesses wider contamination at the partition level. To estimate contamination of individual instances, we employ "guided instruction:" a prompt consisting of the dataset name, partition type, and the random-length initial segment of a reference instance, asking the LLM to complete it. An instance is flagged as contaminated if the LLM\'s output either exactly or nearly matches the latter segment of the reference. To understand if an entire partition is contaminated, we propose two ideas. The first idea marks a dataset partition as contaminated if the average overlap score with the reference instances (as measured by ROUGE-L or BLEURT) is statistically significantly better with the completions from guided instruction compared to a "general instruction" that does not include the dataset and partition name. The second idea marks a dataset partition as contaminated if a classifier based on GPT-4 with few-shot in-context learning prompt marks multiple generated completions as exact/near-exact matches of the corresponding reference instances. Our best method achieves an accuracy between 92% and 100% in detecting if an LLM is contaminated with seven datasets, containing train and test/validation partitions, when contrasted with manual evaluation by human experts. Further, our findings indicate that GPT-4 is contaminated with AG News, WNLI, and XSum datasets.
- **OpenReview**: https://openreview.net/pdf?id=2Rwq6c3tvr
        
</details>

### PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization
> PromptAgent is an innovative technique that leverages strategic planning and intentional exploration to discover expert-level prompts for various language tasks, achieving remarkable results across multiple domains.

<details>
<summary>Details</summary>
- **Abstract**: Expert-level prompts, carefully engineered by human experts who have a deep understanding of both large language models (LLMs) and domain knowledge, are the future of prompting and pivotal to harnessing the full power of advanced LLMs. Discovering such prompts with an automated process remains a sought-after and unresolved challenge. Existing prompt optimization techniques, though automated through iterative sampling, often fall short in injecting domain knowledge and exploring the vast prompt space for complex expert-level prompts efficiently. To address this pressing need and achieve expert-level prompting, we introduce PromptAgent, which autonomously discovers prompts equivalent in quality to those handcrafted by experts. At its core, PromptAgent views prompt optimization as a strategic planning problem and employs a principled planning algorithm (rooted in Monte Carlo Tree Search) to strategically explore the vast expert-level prompt space. PromptAgent interacts with the LLM in a human-like trial-and-error manner during the planning, and injects expert-level knowledge by reflecting on model errors and generating insightful error feedback. This novel formulation allows it to iteratively evaluate intermediate prompts, refine them based on errors, simulate future rewards, and search for high-reward paths leading to expert-level prompts. We apply PromptAgent to 12 tasks spanning three practical domains: BIG-Bench Hard (BBH), domain-expert, and general NLU tasks, showing PromptAgent consistently outperforms strong prompting and prompt optimization baselines by great margins. Our qualitative analysis further emphasizes PromptAgent's capability to distill insightful errors into expert-level prompts.
- **OpenReview**: https://openreview.net/pdf?id=22pyNMuIoa
        
</details>

### MUFFIN: Curating Multi-Faceted Instructions for Improving Instruction Following
> MUFFIN presents a novel data curation approach for enhancing instruction-following capabilities of large language models (LLMs), which addresses limitations in existing schemes by combining diverse input facets and expanding task variety to foster better adherence to instructions.

<details>
<summary>Details</summary>
- **Abstract**: In the realm of large language models (LLMs), enhancing instruction-following capability often involves curating expansive training data. This is achieved through two primary schemes: i) Scaling-Inputs: Amplifying (input, output) pairs per task instruction, aiming for better instruction adherence. ii) Scaling Input-Free Tasks: Enlarging tasks, each composed of an (instruction, output) pair (without requiring a separate input anymore). However, LLMs under Scaling-Inputs tend to be overly sensitive to inputs, leading to misinterpretation or non-compliance with instructions. Conversely, Scaling Input-Free Tasks demands a substantial number of tasks but is less effective in instruction following when dealing with instances in Scaling-Inputs. This work introduces MUFFIN, a new scheme of instruction-following dataset curation. Specifically, we automatically Scale Tasks per Input by diversifying these tasks with various input facets. Experimental results across four zero-shot benchmarks, spanning both Scaling-Inputs and Scaling Input-Free Tasks schemes, reveal that LLMs, at various scales, trained on MUFFIN generally demonstrate superior instruction-following capabilities compared to those trained on the two aforementioned schemes.
- **OpenReview**: https://openreview.net/pdf?id=1vrS1zwekw
        
</details>

### Unified Human-Scene Interaction via Prompted Chain-of-Contacts
> UniHSI, a novel framework for Human-Scene Interaction, utilizes language commands to control diverse interactions through a unified Chain of Contacts representation, enabling versatile task execution and generalizability across various scenes.

<details>
<summary>Details</summary>
- **Abstract**: Human-Scene Interaction (HSI) is a vital component of fields like embodied AI and virtual reality. Despite advancements in motion quality and physical plausibility, two pivotal factors, versatile interaction control and the development of a user-friendly interface, require further exploration before the practical application of HSI. This paper presents a unified HSI framework, UniHSI, which supports unified control of diverse interactions through language commands. This framework is built upon the definition of interaction as Chain of Contacts (CoC): steps of human joint-object part pairs, which is inspired by the strong correlation between interaction types and human-object contact regions. Based on the definition, UniHSI constitutes a Large Language Model (LLM) Planner to translate language prompts into task plans in the form of CoC, and a Unified Controller that turns CoC into uniform task execution. To facilitate training and evaluation, we collect a new dataset named ScenePlan that encompasses thousands of task plans generated by LLMs based on diverse scenarios. Comprehensive experiments demonstrate the effectiveness of our framework in versatile task execution and generalizability to real scanned scenes.
- **OpenReview**: https://openreview.net/pdf?id=1vCnDyQkjg
        
</details>

### Dynamic Sparse No Training:  Training-Free Fine-tuning for Sparse LLMs
> Dynamic Sparse No Training ($\\texttt{DSNT}$) bridges the gap in fine-tuning large language models (LLMs) by enabling training-free updates of sparse models, resulting in enhanced performance without the computational burden of fine-tuning.

<details>
<summary>Details</summary>
- **Abstract**: The ever-increasing large language models (LLMs), though opening a potential path for the upcoming artificial general intelligence, sadly drops a daunting obstacle on the way towards their on-device deployment. As one of the most well-established pre-LLMs approaches in reducing model complexity, network pruning appears to lag behind in the era of LLMs, due mostly to its costly fine-tuning (or re-training) necessity under the massive volumes of model parameter and training data. To close this industry-academia gap, we introduce Dynamic Sparse No Training ($\\texttt{DSNT}$), a training-free fine-tuning approach that slightly updates sparse LLMs without the expensive backpropagation and any weight updates. Inspired by the Dynamic Sparse Training, $\\texttt{DSNT}$ minimizes the reconstruction error between the dense and sparse LLMs, in the fashion of performing iterative weight pruning-and-growing on top of sparse LLMs. To accomplish this purpose, $\\texttt{DSNT}$ particularly takes into account the anticipated reduction in reconstruction error for pruning and growing, as well as the variance w.r.t. different input data for growing each weight. This practice can be executed efficiently in linear time since its obviates the need of backpropagation for fine-tuning LLMs. Extensive experiments on LLaMA-V1/V2, Vicuna, and OPT across various benchmarks demonstrate the effectiveness of $\\texttt{DSNT}$ in enhancing the performance of sparse LLMs, especially at high sparsity levels. For instance, $\\texttt{DSNT}$ is able to outperform the state-of-the-art Wanda by 26.79 perplexity at 70% sparsity with LLaMA-7B. Our paper offers fresh insights into how to fine-tune sparse LLMs in an efficient training-free manner and open new venues to scale the great potential of sparsity to LLMs. Codes will be released.
- **OpenReview**: https://openreview.net/pdf?id=1ndDmZdT4g
        
</details>

### Q-Bench: A Benchmark for General-Purpose Foundation Models on Low-level Vision
> Q-Bench, a benchmark to assess Multi-modality Large Language Models' (MLLMs) emerging low-level visual perception and description abilities, reveals their potential but also highlights the need for further development in these areas.

<details>
<summary>Details</summary>
- **Abstract**: The rapid evolution of Multi-modality Large Language Models (MLLMs) has catalyzed a shift in computer vision from specialized models to general-purpose foundation models. Nevertheless, there is still an inadequacy in assessing the abilities of MLLMs on **low-level visual perception and understanding**. To address this gap, we present **Q-Bench**, a holistic benchmark crafted to systematically evaluate potential abilities of MLLMs on three realms: low-level visual perception, low-level visual description, and overall visual quality assessment. **_a)_** To evaluate the low-level **_perception_** ability, we construct the **LLVisionQA** dataset, consisting of 2,990 diverse-sourced images, each equipped with a human-asked question focusing on its low-level attributes. We then measure the correctness of MLLMs on answering these questions. **_b)_** To examine the **_description_** ability of MLLMs on low-level information, we propose the **LLDescribe** dataset consisting of long expert-labelled *golden* low-level text descriptions on 499 images, and a GPT-involved comparison pipeline between outputs of MLLMs and the *golden* descriptions. **_c)_** Besides these two tasks, we further measure their visual quality **_assessment_** ability to align with human opinion scores. Specifically, we design a softmax-based strategy that enables MLLMs to predict *quantifiable* quality scores, and evaluate them on various existing image quality assessment (IQA) datasets. Our evaluation across the three abilities confirms that MLLMs possess preliminary low-level visual skills. However, these skills are still unstable and relatively imprecise, indicating the need for specific enhancements on MLLMs towards these abilities. We hope that our benchmark can encourage the research community to delve deeper to discover and enhance these untapped potentials of MLLMs.
- **OpenReview**: https://openreview.net/pdf?id=0V5TVt9bk0
        
</details>

### Planting a SEED of Vision in Large Language Model
> SEED, an image tokenizer, enables LLMs to "SEE" and "D"raw by representing images as interchangeable tokens within the LLM\'s autoregressive framework, facilitating the emergence of multimodal comprehension and generation abilities, including compositional in-context generation, a step towards AGI.

<details>
<summary>Details</summary>
- **Abstract**: The great success of Large Language Models (LLMs) has expanded the potential of multimodality, contributing to the gradual evolution of General Artificial Intelligence (AGI). A true AGI agent should not only possess the capability to perform predefined multi-tasks but also exhibit emergent abilities in an open-world context. However, despite the considerable advancements made by recent multimodal LLMs, they still fall short in effectively unifying comprehension and generation tasks, let alone open-world emergent abilities. We contend that the key to overcoming the present impasse lies in enabling text and images to be represented and processed interchangeably within a unified autoregressive Transformer. To this end, we introduce $\\textbf{SEED}$, an elaborate image tokenizer that empowers LLMs with the ability to $\\textbf{SEE}$ and $\\textbf{D}$raw at the same time. We identify two crucial design principles: (1) Image tokens should be independent of 2D physical patch positions and instead be produced with a $\\textit{1D causal dependency}$, exhibiting intrinsic interdependence that aligns with the left-to-right autoregressive prediction mechanism in LLMs. (2) Image tokens should capture $\\textit{high-level semantics}$ consistent with the degree of semantic abstraction in words, and be optimized for both discriminativeness and reconstruction during the tokenizer training phase. With SEED tokens, LLM is able to perform scalable multimodal autoregression under its original training recipe, i.e., next-word prediction. SEED-LLaMA is therefore produced by large-scale pretraining and instruction tuning on the interleaved textual and visual data, demonstrating impressive performance on a broad range of multimodal comprehension and generation tasks. More importantly, SEED-LLaMA has exhibited compositional emergent abilities such as multi-turn in-context multimodal generation, acting like your AI assistant. The code and models will be publicly released.
- **OpenReview**: https://openreview.net/pdf?id=0Nui91LBQS
        
</details>

### Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning
> This paper proposes a cost-efficient method for tailoring smaller yet powerful LLMs from large pre-trained counterparts using targeted structured pruning and dynamic batch loading, resulting in significant computational savings while maintaining or surpassing the performance of competitive open-source models.

<details>
<summary>Details</summary>
- **Abstract**: The popularity of LLaMA (Touvron et al., 2023a;b) and other recently emerged moderate-sized large language models (LLMs) highlights the potential of building smaller yet powerful LLMs. Regardless, the cost of training such models from scratch on trillions of tokens remains high. In this work, we study structured pruning as an effective means to develop smaller LLMs from pre-trained, larger models. Our approach employs two key techniques: (1) targeted structured pruning, which prunes a larger model to a specified target shape by removing layers, heads, intermediate and hidden dimensions in an end-to-end manner, and (2) dynamic batch loading, which dynamically updates the composition of sampled data in each training batch based on varying losses across different domains. We demonstrate the efficacy of our approach by presenting the Sheared-LLaMA series, pruning the LLaMA2-7B model down to 1.3B and 2.7B parameters. Sheared-LLaMA models outperform state-of-the-art open-source models of equivalent sizes, such as Pythia, INCITE, and OpenLLaMA models, on a wide range of downstream and instruction tuning evaluations, while requiring less than 3% of compute compared to training such models from scratch. This work provides compelling evidence that leveraging existing LLMs with structured pruning is a far more cost-effective approach for building smaller LLMs.
- **OpenReview**: https://openreview.net/pdf?id=09iOdaeOzp
        
</details>

### Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing
> In this study, the authors investigate a smart inference approach for natural language processing tasks, proposing a hybrid model that combines a small, efficient model with a larger, more powerful one. The approach dynamically assigns queries to the appropriate model based on their complexity and the desired quality level, potentially reducing the number of calls to the large model by up to 40% without compromising response quality. This cost-saving solution allows for flexible quality-cost trade-offs in real-time scenarios.

<details>
<summary>Details</summary>
- **Abstract**: Large language models (LLMs) excel in most NLP tasks but also require expensive cloud servers for deployment due to their size, while smaller models that can be deployed on lower cost (e.g., edge) devices, tend to lag behind in terms of response quality. Therefore in this work we propose a hybrid inference approach which combines their respective strengths to save cost and maintain quality. Our approach uses a router that assigns queries to the small or large model based on the predicted query difficulty and the desired quality level. The desired quality level can be tuned dynamically at test time to seamlessly trade  quality for cost as per the scenario requirements. In experiments our approach allows us to make up to 40% fewer calls to the large model, with no drop in response quality.
- **OpenReview**: https://openreview.net/pdf?id=02f3mUtqnM
        
</details>

