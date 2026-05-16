### How the algorithm works

The algorithm is a constrained breadth-first tree growth over real 3D stellar
positions from the AT-HYG catalog in `athyg-33.csv`. It begins at Sol and
grows outward until exactly 268 leaf entities each occupy a unique destination
star.

Spectral ranking is the backbone of the constraint system. Every star's
Morgan-Keenan class (O B A F G K M, from hottest to coolest) and numeric
subclass are converted to a single integer rank. Sol as a G2V gets rank 27,
while an O0 star would be 69 and an M9 would be 0. The invariant is that no
entity may ever travel to a star with a lower rank than its inherited floor.
At each fork, children independently decide whether to keep their parent's
floor or raise it by one step, so some lineages stay near Sol's class while
others drift toward hotter stars.

Destination selection uses a KD-tree for fast spatial queries combined with a
dual scoring function. For each entity, the algorithm queries the nearest stars
in expanding batches, filters to stars above the rank floor and not yet
claimed, then scores each candidate as a weighted blend of proximity and
directional alignment. This produces a biased random walk: entities are pulled
toward nearby stars but steered by their travel direction.

Edge non-intersection is handled heuristically through directional divergence.
The first fork at the initial destination sends two children in exactly
opposing directions. Every subsequent fork distributes children in a cone
around the parent's incoming direction, spaced at equal azimuthal angles with
jitter. Because siblings are pointed away from each other and each entity keeps
moving roughly outward, paths naturally fan apart in 3D space.

Branching and leaf counting follow BFS ordering. When a leaf is processed, it
spawns 1-3 children chosen uniformly, clamped so the leaf total never overshoots
268. The parent stops being a leaf (-1) and its `b` children become new leaves
(+b), so each fork adds `(b - 1)` net leaves.

### Output format

The output is written as TOML with the following structure:

- `metadata`
- `origin`
- `tree`
- `leaf_destinations`

The default output filename is versioned automatically. Each run without an
explicit `--output` writes the next available file in the current working
directory:

- `stellar_tree_version1.toml`
- `stellar_tree_version2.toml`
- `stellar_tree_version3.toml`

If you pass `--output`, that explicit path is used instead.

### Run commands

Run the script with the repo-local catalog:

```bash
python ./journey_algo.py
```

Run with an explicit seed:

```bash
python ./journey_algo.py --seed 42
```

Run with an explicit target leaf count:

```bash
python ./journey_algo.py --target 268
```

Run with an explicit output path:

```bash
python ./journey_algo.py --output ./stellar_tree_custom.toml
```
