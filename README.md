# **Diffusion Lens: Interpreting Text Encoders in Text-to-Image pipelines**

**Authors: Ivan Golov, Roman Makeev**
---

## **Introduction**

In this work, we introduce an interpretable, end-to-end framework that enhances **Stable Diffusion v1.5 model** fine‑tuned via the [DreamBooth method](https://dreambooth.github.io) [1] to generate high‑fidelity, subject‑driven images from as few reference examples. 

While DreamBooth effectively personalizes generation by associating a unique rare token with the target concept, the internal process through which textual prompts are transformed into visual representations remains opaque. To bridge this gap, we integrate [Diffusion Lens](https://tokeron.github.io/DiffusionLensWeb/) [2], a visualization technique that decodes the text encoder’s intermediate hidden states into images, producing a layer‑by‑layer sequence that illuminates how semantic concepts emerge and refine over the course of encoding.

**By uniting DreamBooth’s subject‑specific fine‑tuning with Diffusion Lens’s interpretability during both training and inference, our framework not only delivers compelling, stylized outputs but also offers transparent, quantitative insights into the hierarchical construction of visual content from natural language descriptions.**

## **Background**

### **Section 1: DreamBooth Fine-Tuning**

DreamBooth [1] fine-tunes a pre-trained diffusion model **with a small set (3–5) of images of a subject by binding a unique, rare-token identifier to the subject**. The rare token, chosen from the text encoder’s vocabulary, acts as a minimal prior and is used to encode target image features and styles. The main training objective is given by:

![Loss functions](/static/dreambooth_math.png)

An additional prior preservation loss ensures that the model retains its generalization over the subject’s class even after fine-tuning.  

![DreamBooth framework](/static/dreambooth.png)

Figure 1: Illustration of the DreamBooth approach: Fine-tuning the diffusion model using rare tokens to
encode target subject details and style

![DreamBooth example](/static/dreambooth_examples.png)

Figure 2: Expected output from DreamBooth fine-tuning: Images generated that exhibit the target subject
details and stylistic features as encoded by the rare tokens.

### **Section 2: Diffusion Lens Interpretability**

Diffusion Lens [2] is employed to analyze **the internal representations of the text encoder after the fine-tuning process**. Rather than solely relying on the final output, we generate images from intermediate hidden states.
For a given layer l (with l < L for a total of L layers), the generated image is:

![Diffusion Lens math](/static/DiffLen_math.png)

This method provides:
* **Layer-by-Layer Understanding:** Early layers capture basic, unstructured representations (a “bag
of concepts”), while later layers progressively refine and organize these ideas.
* **Complexity Analysis:** Simple prompts (e.g., “a cat”) yield clear representations in early layers,
whereas complex prompts (e.g., “a red car next to a blue bike”) require deeper layers to form accurate
relational structures.
* **Concept Frequency Insights:** Common concepts appear early; uncommon or detailed concepts
emerge only in higher layers.
* **Impact Analysis:** By comparing the intermediate representations before and after applying adapter
techniques (e.g., LoRA), we can study how such modifications alter the text encoder’s understanding
and the final image generation.
* **No Extra Training Required:** The analysis leverages the pre-trained model without modifying its
architecture

![Diffusion Lens Diagram](/static/difflens.png)

## **Methodology**

We investigate the fine‑tuning and inference phases to elucidate how Stable Diffusion v1.5 internalizes and expresses subject‑specific concepts:

1. **Intermediate Diffusion Outputs:**
We set up the U‑Net denoising pipeline to capture latent representations at selected epochs. By visualizing these snapshots, we observe how the model gradually injecting the target class’s distinctive features (shape, texture, lighting) learned via DreamBooth.

2. **Text Encoder Layer Visualization:**
During prompt encoding, we capture the hidden states {{<katex>}}h_\ell{{</katex>}} at multiple encoder layers. Each extracted representation is decoded through the frozen diffusion decoder, yielding images that reveal the semantic content captured at that stage. This layer‑wise decoding clarifies when and to what extent the rare token’s semantics integrate with the overall prompt, illuminating the encoder’s hierarchical concept construction.

### **Implementation Highlights**

Our codebase is organized as follows:

```
GAI_course_project/
├── analyze_dreambooth_lens.py    # Script for analyzing the codebase provide by DiffusionLens framework.
├── configs/                      # Directory containing configuration files for different experiments.
├── data/                         # Directory containing the data used for training.
├── DiffusionLens/                # Directory related to the DiffusionLens module.
├── images/                       # Directory containing images for reporting.
├── inference_outputs/            # Directory to store the outputs of the inference process.
├── lens_output/                  # Directory to store the output specifically related to Diffusion Lens experiments.
├── LICENSE                       # License file for the project.
├── notebooks/                    # Directory containing Jupyter notebooks for experimentation and analysis.
├── outputs/                      # Directory to store outputs of traning and inference pipelines - images, stats, models weights.
├── papers/                        # Directory related to a research papers and supplementary materials.
├── poetry.lock                   # Lock file for dependency management using Poetry.
├── pyproject.toml                # Project configuration file for Poetry.
├── README.md                     # Project description and instructions.
├── requirements.txt              # List of project dependencies.
└── scripts/                      # Directory containing utility scripts.
└── src/                          # Directory containing the main source code of the project.
```




## **Experiments and Analysis**


## **Conclusion**

By combining DreamBooth fine-tuning with Diffusion Lens interpretability, we achieve not only **high-fidelity, subject-driven image synthesis** but also **transparent insights** into the model’s inner semantic processing. Our visualizations confirm that concepts emerge and sharpen progressively across text encoder and U-net layers.  


## **References**

[1] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman, Dreambooth: Fine tuning text-
to-image diffusion models for subject-driven generation, 2023. arXiv: 2208.12242 [cs.CV]. [Online]. Available: [https://arxiv.org/abs/2208.12242](https://arxiv.org/abs/2208.12242).

[2] M. Toker, H. Orgad, M. Ventura, D. Arad, and Y. Belinkov, “Diffusion lens: Interpreting text encoders in text-to-image pipelines,” Association for Computational Linguistics, 2024, pp. 9713–9728. doi: 10.18653/v1/2024.acl-long.524. [Online]. Available: [http://dx.doi.org/10.18653/v1/2024.acl-long.524](https://arxiv.org/abs/2208.12242).


