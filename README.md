# Impact Cipher dataset source

Impact Cipher is an original synthetic benchmark of ten-disc elastic collision systems. This repository publishes the numerical generator, dataset schema, CC0 dedication, dependency versions, and release hashes. It does not contain the raw corpus, private answers, hidden masses, or the organizer release seed.

Generate an independent corpus with Python 3.11+:

```bash
python generate_dataset.py --out raw_dataset --seed YOUR_INTEGER_SEED --workers 8
```

Any integer seed creates an independent reproducible corpus. The organizer retains the original release seed because source-plus-seed replay would reveal private labels. `generation_metadata.json` publishes a SHA-256 commitment to that seed and the raw release hash.

No external dataset, personal data, downloaded asset, copyrighted media, or pretrained-model output is used. Code and documentation are AI-assisted. The simulator is a deterministic discrete hard-disc benchmark with analytic unit-restitution pair impulses; it is not a claim of continuous-time simultaneous-contact accuracy.
