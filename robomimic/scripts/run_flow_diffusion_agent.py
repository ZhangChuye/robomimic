"""Backward-compatible combined entrypoint.

Prefer:
  python run_flow_diffusion_agent_gt.py ...
  python run_flow_diffusion_agent_generated.py ...
"""

from flow_diffusion_agent_common import main


if __name__ == "__main__":
    main()
