"""Test the user's proposed style families against the measurements.

Space = standardized classic measurements (minus warmth, which mostly tracks
the animal's own fur color) + standardized style fingerprint (Gram PCA),
each block weighted equally.

For each family:
  - tightness: how close members sit vs. the collection's typical spread
  - leave-one-out: does each member land nearest its own family?
  - defining traits: measurements where the family sits far from typical
  - nearest pieces: every other master ranked by distance to the family

Writes art-pipeline/vetting/style_families.json (consumed by axes.html).
Usage: art-pipeline/funnel-venv/Scripts/python art-pipeline/style_families.py
"""

import json
from pathlib import Path

import numpy as np

VET = Path(__file__).parent / "vetting"
EXCLUDE = {"warmth"}


def main() -> None:
    D = json.loads((VET / "style_axes.json").read_text())
    U = json.loads((VET / "style_families_user.json").read_text())
    pieces = D["pieces"]
    slugs = [p["slug"] for p in pieces]
    idx = {s: i for i, s in enumerate(slugs)}
    feats = [f for f in D["features"] if f not in EXCLUDE]

    A = np.array([[p[f] for f in feats] for p in pieces], dtype=float)
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    G = np.array([p["gram_pca"] for p in pieces], dtype=float)
    G = (G - G.mean(0)) / (G.std(0) + 1e-9)
    X = np.hstack([A / np.sqrt(A.shape[1]), G / np.sqrt(G.shape[1])])
    typical = float(np.median(np.linalg.norm(X[:, None] - X[None], axis=-1)))

    fams = {k: [m for m in v["members"] if m in idx] for k, v in U["families"].items()}
    cent = {k: X[[idx[m] for m in ms]].mean(0) for k, ms in fams.items()}

    out = {"typical_distance": round(typical, 3), "families": {}}
    for k, ms in fams.items():
        M = X[[idx[m] for m in ms]]
        tight = float(np.linalg.norm(M[:, None] - M[None], axis=-1)[np.triu_indices(len(ms), 1)].mean()) if len(ms) > 1 else None

        loo = {}
        for m in ms:
            others = [o for o in ms if o != m]
            if not others:
                continue
            c_own = X[[idx[o] for o in others]].mean(0)
            d_own = float(np.linalg.norm(X[idx[m]] - c_own))
            d_rest = {k2: float(np.linalg.norm(X[idx[m]] - c)) for k2, c in cent.items() if k2 != k}
            best_other = min(d_rest, key=d_rest.get)
            loo[m] = {"own": round(d_own, 3), "nearest_other": best_other,
                      "nearest_other_dist": round(d_rest[best_other], 3), "stays": d_own < d_rest[best_other]}

        fam_med = np.median(A[[idx[m] for m in ms]], 0)
        traits = sorted([(feats[j], round(float(fam_med[j]), 2)) for j in range(len(feats))
                         if abs(fam_med[j]) >= 0.7], key=lambda t: -abs(t[1]))

        dists = np.linalg.norm(X - cent[k], axis=1)
        nearest = [{"slug": slugs[i], "dist": round(float(dists[i]), 3), "tier": pieces[i]["tier"]}
                   for i in np.argsort(dists) if slugs[i] not in ms][:12]

        out["families"][k] = {"label": U["families"][k]["label"], "verdict": U["families"][k]["verdict"],
                              "members": ms, "tightness": None if tight is None else round(tight, 3),
                              "leave_one_out": loo, "traits": traits, "nearest": nearest}

    # where each unplaced like / every piece lands
    assign = {}
    for i, s in enumerate(slugs):
        d = sorted(((k, float(np.linalg.norm(X[i] - c))) for k, c in cent.items()), key=lambda t: t[1])
        assign[s] = {"family": d[0][0], "dist": round(d[0][1], 3), "runner_up": d[1][0],
                     "margin": round((d[1][1] - d[0][1]) / max(d[0][1], 1e-6), 3)}
    out["assignment"] = assign
    (VET / "style_families.json").write_text(json.dumps(out, indent=1))

    print(f"typical distance between any two pieces: {typical:.2f}\n")
    for k, f in out["families"].items():
        t = f"tightness {f['tightness']:.2f} ({f['tightness']/typical:.0%} of typical)" if f["tightness"] else "single anchor"
        stays = sum(v["stays"] for v in f["leave_one_out"].values())
        print(f"[{k}] {t} | leave-one-out {stays}/{len(f['leave_one_out'])} stay home")
        print(f"   traits: {', '.join(f'{n} {v:+.1f}' for n, v in f['traits'][:6]) or '(nothing distinctive)'}")
        print(f"   nearest: {', '.join(n['slug'] for n in f['nearest'][:8])}")
    print("\nunplaced likes land in:")
    for s in U["unplaced_likes"]:
        a = assign[s]
        print(f"   {s:22s} -> {a['family']} (runner-up {a['runner_up']}, margin {a['margin']:.0%})")


if __name__ == "__main__":
    main()
