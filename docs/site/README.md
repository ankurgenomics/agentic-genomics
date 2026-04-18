# Portfolio site — `docs/site/`

Static portfolio for Ankur Sharma (ankurgenomics), deployed via **GitHub Pages**.

## Files

- `index.html` — the page
- `styles.css` — all styles (dark, modern, no framework)
- `assets/` — OG image, resume PDF (when added)

## Preview locally

No build step. Any static server works:

```bash
# Option 1: Python's built-in server
cd docs/site && python -m http.server 8000
# then open http://localhost:8000

# Option 2: VS Code's "Live Server" extension — right-click index.html → Open with Live Server
```

## Deployment (GitHub Pages)

The workflow at `.github/workflows/pages.yml` auto-deploys `docs/site/` on every push to `main`.

**One-time setup on GitHub.com:**

1. Push the repo to `github.com/ankurgenomics/agentic-genomics`.
2. Go to **Settings → Pages**.
3. Under *Source*, select **GitHub Actions** (not "Deploy from a branch").
4. Push any change under `docs/site/` — the workflow will run and publish the site.
5. The live URL is typically `https://ankurgenomics.github.io/agentic-genomics/`.

For a custom domain later (e.g. `ankursharma.dev`): add a `CNAME` file in `docs/site/` containing just the domain, and configure DNS.

## Things to fill in before sharing widely

- [ ] Add `assets/og-image.png` (1200×630) — used for link previews on LinkedIn/Twitter/Slack. A single-screen screenshot of the hero works well.
- [ ] Add `assets/resume.pdf` and link it from the hero (`<a class="btn btn-ghost" href="assets/resume.pdf">Resume</a>`).
- [ ] Verify all project links (`GitHub repo`, `Why agentic?`, `Architecture`) resolve after the main repo is pushed public.
- [ ] Double-check `ankurs103@gmail.com` is the right contact email (or change it).

## Content rules of thumb

- Every project card should have at least **one** working link (code, writeup, demo, or an explicit "internal" disclaimer).
- Metrics must be real and defensible in an interview — don't inflate.
- The hero headline should match the job titles you're actually applying to.
