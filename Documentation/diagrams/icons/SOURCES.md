# Logo sources

Logos used in [`architecture-slides.tex`](../architecture-slides.tex). Vector logos are stored as the original SVG
and as a PDF conversion (for `\includegraphics`); raster logos are PNG. All logos are trademarks of their
respective owners and are used here only to identify the tools in the architecture diagram.

| File | Tool | Source | License of the file |
|---|---|---|---|
| `deepcncf.png` | DeepCNCF (team logo) | [Deliverables/sprint-01/team-logo.png](../../../Deliverables/sprint-01/team-logo.png) | this project (MIT) |
| `gemma.png` | Gemma | Gemma artwork supplied by the project team, cropped to a circle | Google brand asset |
| `langchain` | LangChain | Logo file (parrot + chain) supplied by the project team (`langchain-seeklogo.svg`) | LangChain brand asset |
| `llamacpp-mark` | llama.cpp | `cover-llama-cpp-dark.svg` supplied by the project team, cropped to the shape only (viewBox `2333.5 970 560 560`), shown on a dark badge | llama.cpp brand asset (MIT-licensed project) |
| `refresh` | "re-run periodically" symbol | [Lucide](https://lucide.dev/icons/refresh-cw) `refresh-cw` (lucide-static 0.469.0), recoloured | ISC |
| `cncf` | Cloud Native Computing Foundation | [Simple Icons](https://simpleicons.org/?q=cncf) (`cdn.simpleicons.org/cncf`) | CC0 1.0 |
| `stackoverflow` | Stack Overflow | [Simple Icons](https://simpleicons.org/?q=stackoverflow) | CC0 1.0 |
| `github` | GitHub | [Simple Icons](https://simpleicons.org/?q=github) | CC0 1.0 |
| `scrapy` | Scrapy | [Simple Icons](https://simpleicons.org/?q=scrapy) | CC0 1.0 |
| `python` | Python | [Simple Icons](https://simpleicons.org/?q=python) | CC0 1.0 |
| `huggingface` | Hugging Face | [huggingface.co/front/assets/huggingface_logo-noborder.svg](https://huggingface.co/front/assets/huggingface_logo-noborder.svg) | Hugging Face brand asset |
| `chroma` | Chroma | [chroma-core/chroma `docs/mintlify/images/favicon.svg`](https://github.com/chroma-core/chroma/blob/main/docs/mintlify/images/favicon.svg) | from Apache-2.0-licensed repo |

SVG → PDF conversion: [`svglib`](https://pypi.org/project/svglib/) (`svg2rlg` + `renderPDF.drawToFile`).
