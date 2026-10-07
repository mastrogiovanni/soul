# Soul community website

Public website: **https://mastrogiovanni.github.io/soul-community/**

The source of truth is `site/` in this repository. It is a standalone static site with no build dependencies, tracking, forms, or embedded credentials. Preview it with:

```sh
python3 -m http.server 4173 --directory site
```

Open `http://localhost:4173`. The gallery supports keyboard navigation and Escape; copy buttons select the command when clipboard permission is unavailable. Installation, inference instructions, and links also work without JavaScript. Layouts support small screens and reduced-motion preferences.

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

On 7 October 2026, `/soul/healthz` returned 200 but `/soul/install.sh` returned 404. The join panel displays this deployment dependency explicitly. Remove the `.installer-status` notice after the installer and archive endpoints are deployed and verified. Publishing Pages does not update the separately hosted controller.

- `site/assets/network.jpg`: generated conceptual network artwork, used in the hero, gallery, and social preview.
- `site/assets/workspace.jpg`: generated illustrative contributor workspace.
- `site/assets/console.jpg`: copied from `docs/admin-console.jpg`, showing local demo data, not live statistics.
- `site/assets/favicon.svg`: code-native version of the existing Soul star motif.

Raster images were created with the built-in image generation tool and converted to JPEG for delivery. They are hosted locally with the site; no stock-photo service is required. See `docs/website-image-prompts.txt` for the full prompts.

## Verification

Before publishing, inspect desktop and mobile layouts, follow the section links, open every gallery image, verify next/previous/Escape controls, copy the install and API commands, expand FAQ/installation details, and check for horizontal overflow and browser errors. Check the public installer and controller links independently; publishing this static site does not deploy the controller.
