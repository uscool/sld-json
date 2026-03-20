"""
Main service for generating single line diagrams.

Orchestrates two diagram generators:
  - Standalone: one topology per diagram, all six topology types
  - Hybrid:     multi-voltage-level stacks of mixed topologies

Mirrors the interface of the legacy DiagramService so it can act as a
drop-in orchestrator replacement on this branch.

Usage
-----
# Default run — standalone + hybrid with built-in sample counts
python diagram_service.py

# Bulk mode — total diagrams split evenly between standalone and hybrid
python diagram_service.py --bulk-count 500

# Fine-grained control
python diagram_service.py --standalone-samples 100 --hybrid-samples 200

# Override output directories
python diagram_service.py --standalone-dir output/standalone --hybrid-dir output/hybrid
"""

import argparse
import sys, os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import core.main_standalone as standalone
import core.main_hybrid as hybrid


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIG  (mirrors the globals in each sub-script)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STANDALONE_DIR     = "output_standalone"
DEFAULT_HYBRID_DIR         = "output_hybrid"
DEFAULT_STANDALONE_SAMPLES = standalone.NUM_SAMPLES   # per topology
DEFAULT_HYBRID_SAMPLES     = hybrid.NUM_SAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR SERVICE
# ─────────────────────────────────────────────────────────────────────────────

class DiagramService:
    """
    Orchestrates standalone and hybrid diagram generation.

    Parameters
    ----------
    standalone_dir : str
        Output directory for standalone diagrams.
    hybrid_dir : str
        Output directory for hybrid diagrams.
    standalone_samples : int
        Number of samples to generate *per topology* in standalone mode.
    hybrid_samples : int
        Total number of hybrid multi-level diagrams to generate.
    """

    def __init__(
        self,
        standalone_dir:     str = DEFAULT_STANDALONE_DIR,
        hybrid_dir:         str = DEFAULT_HYBRID_DIR,
        standalone_samples: int = DEFAULT_STANDALONE_SAMPLES,
        hybrid_samples:     int = DEFAULT_HYBRID_SAMPLES,
    ):
        self.standalone_dir     = standalone_dir
        self.hybrid_dir         = hybrid_dir
        self.standalone_samples = standalone_samples
        self.hybrid_samples     = hybrid_samples

        os.makedirs(standalone_dir, exist_ok=True)
        os.makedirs(hybrid_dir,     exist_ok=True)

    # ── Individual generators ─────────────────────────────────────────────────

    def generate_standalone(self):
        """
        Generate standalone diagrams — NUM_SAMPLES per topology type
        (SINGLE_BUS, DOUBLE_BUS, BREAKER_AND_HALF, MAIN_AND_TRANSFER,
        RING_BUS, MESH).
        """
        num_topologies = len(standalone.RENDER_MAP)
        total = num_topologies * self.standalone_samples
        print(f"\n[Standalone] Generating {self.standalone_samples} samples "
              f"× {num_topologies} topologies = {total} diagrams "
              f"→ {self.standalone_dir}/")
        print("-" * 60)
        standalone.main(
            output_dir  = self.standalone_dir,
            num_samples = self.standalone_samples,
        )

    def generate_hybrid(self):
        """
        Generate hybrid multi-voltage-level diagrams (2–5 levels per diagram,
        topologies chosen randomly per level).
        """
        print(f"\n[Hybrid] Generating {self.hybrid_samples} diagrams "
              f"→ {self.hybrid_dir}/")
        print("-" * 60)
        hybrid.main(
            output_dir  = self.hybrid_dir,
            num_samples = self.hybrid_samples,
        )

    # ── Combined runners ──────────────────────────────────────────────────────

    def generate_all(self):
        """Run both standalone and hybrid generators with current settings."""
        print("=" * 60)
        print("DIAGRAM SERVICE — generating all diagram types")
        print("=" * 60)
        self.generate_standalone()
        self.generate_hybrid()
        print("\n" + "=" * 60)
        print("All generation complete.")
        print(f"  Standalone → {self.standalone_dir}/")
        print(f"  Hybrid     → {self.hybrid_dir}/")
        print("=" * 60)

    def generate_bulk(self, total_diagrams: int):
        """
        Generate `total_diagrams` split evenly between standalone (total
        across all topologies) and hybrid.

        The standalone count is divided across all 6 topology types, so
        standalone_samples = (total // 2) // num_topologies.
        Hybrid gets the other half directly.

        Parameters
        ----------
        total_diagrams : int
            Total diagrams to produce across both generators.
        """
        num_topologies   = len(standalone.RENDER_MAP)
        half             = total_diagrams // 2
        standalone_total = half
        hybrid_total     = total_diagrams - half

        # standalone_samples is per-topology
        per_topo = max(1, standalone_total // num_topologies)

        print("=" * 60)
        print(f"DIAGRAM SERVICE — bulk mode ({total_diagrams} total)")
        print(f"  Standalone: {per_topo} × {num_topologies} topologies "
              f"= {per_topo * num_topologies} diagrams")
        print(f"  Hybrid:     {hybrid_total} diagrams")
        print("=" * 60)

        self.standalone_samples = per_topo
        self.hybrid_samples     = hybrid_total

        self.generate_standalone()
        self.generate_hybrid()

        print("\n" + "=" * 60)
        print("Bulk generation complete.")
        print(f"  Standalone → {self.standalone_dir}/")
        print(f"  Hybrid     → {self.hybrid_dir}/")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Single Line Diagram Generator Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bulk-count",
        type=int,
        default=None,
        help="Total diagrams to generate, split evenly between standalone and hybrid.",
    )
    parser.add_argument(
        "--standalone-samples",
        type=int,
        default=DEFAULT_STANDALONE_SAMPLES,
        help="Samples per topology type for standalone generation.",
    )
    parser.add_argument(
        "--hybrid-samples",
        type=int,
        default=DEFAULT_HYBRID_SAMPLES,
        help="Total hybrid diagrams to generate.",
    )
    parser.add_argument(
        "--standalone-dir",
        default=DEFAULT_STANDALONE_DIR,
        help="Output directory for standalone diagrams.",
    )
    parser.add_argument(
        "--hybrid-dir",
        default=DEFAULT_HYBRID_DIR,
        help="Output directory for hybrid diagrams.",
    )
    parser.add_argument(
        "--standalone-only",
        action="store_true",
        help="Run only the standalone generator.",
    )
    parser.add_argument(
        "--hybrid-only",
        action="store_true",
        help="Run only the hybrid generator.",
    )

    args = parser.parse_args()

    service = DiagramService(
        standalone_dir     = args.standalone_dir,
        hybrid_dir         = args.hybrid_dir,
        standalone_samples = args.standalone_samples,
        hybrid_samples     = args.hybrid_samples,
    )

    if args.bulk_count is not None:
        service.generate_bulk(total_diagrams=args.bulk_count)
    elif args.standalone_only:
        service.generate_standalone()
    elif args.hybrid_only:
        service.generate_hybrid()
    else:
        service.generate_all()


if __name__ == "__main__":
    main()