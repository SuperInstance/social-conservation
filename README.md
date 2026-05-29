# social-conservation

**Social network conservation analysis — engagement level conservation on stochastic block model graphs via graph Laplacian.**

Analyzes conservation of "engagement level" on social network graphs using the graph Laplacian. Uses stochastic block models to generate realistic community structure. Measures how well engagement is conserved (smooth) across the social graph — high conservation means communities are homogeneous.

## What This Gives You

- **Stochastic block model** — configurable community structure with engagement attributes
- **Conservation of engagement** — smoothness of engagement across the social graph
- **Community detection** — spectral partitioning vs ground truth
- **Bridge detection** — inter-community edges where conservation breaks
- **Attribute influence** — correlation, homophily, influence, and random attribute types

## Quick Start

```bash
pip install numpy networkx matplotlib
python analysis.py
```

## How It Fits

Part of the SuperInstance ecosystem:

- **[persistent-social](https://github.com/SuperInstance/persistent-social)** — Social network TDA in Go
- **social-conservation** — Social network spectral analysis in Python (this repo)

## License

MIT
