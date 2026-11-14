"""Command semantics shared by progress costs and their exposure denominator."""

COMMAND_EPSILON = 1e-4
STALL_PROTOCOL_VERSION = 2


def commanded_motion(command):
  """Numerical zero, not a minimum requested walking speed."""
  return (command[:, :2].norm(dim=-1) > COMMAND_EPSILON,
          command[:, 2].abs() > COMMAND_EPSILON)


def command_changed(command, previous):
  return (command-previous).abs().amax(dim=-1) > COMMAND_EPSILON
