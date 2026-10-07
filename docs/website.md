# Soul community website

Public website: **https://mastrogiovanni.github.io/soul-community/**

The source of truth is `site/` in this repository. It is a standalone static site with no build dependencies, tracking, data collection, or embedded credentials. Preview it with:

```sh
python3 -m http.server 4173 --directory site
```

Open `http://localhost:4173`. The gallery supports keyboard navigation and Escape; copy buttons select the command when clipboard permission is unavailable. Installation, inference instructions, and links also work without JavaScript. Layouts support small screens and reduced-motion preferences.

When changing `styles.css`, update its version query in `site/index.html` so returning visitors fetch the new stylesheet immediately after deployment.

## Publishing

The current GitHub plan does not allow Pages on this private repository. A public, site-only repository, `mastrogiovanni/soul-community`, serves Pages from the root of its `main` branch. The application repository stays private.

Pushing changes to `site/`, `scripts/publish-site.sh`, or `.github/workflows/pages.yml` on this repository's `main` runs **Publish community website**. The workflow mirrors only `site/` into the public repository using `scripts/publish-site.sh`. GitHub Pages then builds the public repository. You can also run the workflow manually with `gh workflow run pages.yml`.

Automatic publishing requires approval to create a dedicated SSH deploy key with write access only to `soul-community`, register its public half on that repository, and save its private half in this repository's Actions secret `SOUL_PAGES_DEPLOY_KEY`. Until that secret exists, the workflow reports that setup is pending and skips publication. No general-purpose account token is copied into Actions. GitHub SSH host keys are obtained from its HTTPS metadata endpoint. The temporary private key is removed after the job.

For an authorized manual publication with your existing GitHub SSH access:

```sh
bash scripts/publish-site.sh
```

Keep credentials, application sources, and private files out of `site/`: that entire directory is public. The publisher preserves deployment history and uses ordinary pushes, never force pushes.

## Content and assets

The mission is to make large language models easier to access for communities and nonprofit organizations, accepting higher, variable latency in exchange for shared infrastructure and less setup effort. The use-case section describes nine potential background workflows, including email processing, invoice analysis, and process improvement. These are integration ideas rather than bundled applications: job scheduling, text extraction, retries, and result storage belong to the calling application. Keep this distinction when updating the copy; Soul does not currently implement a durable job service or Batch API.

The join command points to the community controller's `/soul/install.sh`, which must be deployed alongside `/soul/install/agent.tar.gz`. Node enrollment is separate from an inference API key. Public copy describes recovery opportunities and capacity benefits without promising automatic failover or faster responses: this version has one controller and one active generation.

On 7 October 2026, both `/soul/install.sh` and `/soul/install/agent.tar.gz` returned 200, so the earlier rollout-pending notice was removed. Publishing Pages does not update the separately hosted controller.

- `site/assets/network.jpg`: generated conceptual network artwork, used in the hero, gallery, and social preview.
- `site/assets/workspace.jpg`: generated illustrative contributor workspace.
- `site/assets/console.jpg`: copied from `docs/admin-console.jpg`, showing local demo data, not live statistics.
- `site/assets/favicon.svg`: code-native version of the existing Soul star motif.

Raster images were created with the built-in image generation tool and converted to JPEG for delivery. They are hosted locally with the site; no stock-photo service is required. See `docs/website-image-prompts.txt` for the full prompts.

## Model explorer

The `#models` section is a curated catalog of downloadable candidates, not live controller inventory. It includes nine dense, unquantized text models using supported Llama, Mistral, Qwen2, and Qwen3 adapters. Publisher model-card links, exact parameter counts, architecture, and the checked Hugging Face revision are stored with each HTML row. Metadata was checked on 7 October 2026 using `huggingface_hub.HfApi.model_info` and each pinned `config.json`, without downloading weights. Recheck metadata and compatibility when changing this list. Individual model/hardware combinations have not all been tested.

The calculator runs entirely in the browser. Visitors enter an estimated machine count and average memory budget dedicated to Soul; defaults are an illustrative 4 machines × 8 GiB, not measurements of the community controller. CPU/mixed defaults to 4 bytes per parameter, matching CPU float32; GPU/Metal assumes all participating nodes use 2-byte float16 weights. Actual node dtype overrides may differ.

`site/model-estimator.mjs` uses only this arithmetic:

- Capacity in GiB = machine count × contributed GiB per machine.
- Required GiB = `ceil(parameters × bytes_per_parameter / 1024^3 × 1.25 + 2)`.
- Below the requirement: **More memory needed**. At/above it but with less than 10% spare capacity: **Tight fit**. At least 10% spare: **Likely to fit**.
- Suggested additional machines = `max(0, ceil(required_GiB / GiB_per_machine) - machine_count)`.

The overhead and thresholds are heuristics, not benchmarks or placement guarantees. Estimates are for one model at a time on similarly sized machines. They do not model layer placement, context/KV cache, heterogeneous nodes, quantization, disk, or latency. The UI documents these limits and links to the protected `GET /v1/models` example for actual prepared/executable models. Installing/preparing a catalog model remains an operator action in the controller. No credentials or backend authentication changes are needed.

The catalog and model-card links remain readable without JavaScript; calculator/filter controls are disabled until the module loads. Search, the fit filter, presets, zero contributors, and invalid input states are supported. Run the arithmetic tests with `node --test tests/site/*.test.mjs`; this also runs in CI. Update the cache-version queries on changed JS/CSS assets, including module imports, when publishing changes.

## Verification

Before publishing, inspect desktop and mobile layouts, follow the section links, open every gallery image, verify next/previous/Escape controls, copy the install and API commands, expand FAQ/installation details, and check for horizontal overflow and browser errors. For the model explorer, change both memory inputs and precision, try the presets and filters, check invalid/zero-capacity states, and verify the layout on narrow screens. Check the public installer and controller links independently; publishing this static site does not deploy the controller.
